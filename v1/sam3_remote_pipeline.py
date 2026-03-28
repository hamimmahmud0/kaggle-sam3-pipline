#!/usr/bin/env python3
import argparse
import gc
import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt

import numpy as np
import torch
from pycocotools import mask as mask_utils
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

WORKSPACE_ROOT = Path("/kaggle/working/SAM3")
LOGS_DIR = WORKSPACE_ROOT / "logs"
TMP_DIR = WORKSPACE_ROOT / "tmp"
RESULTS_LOCAL_DIR = WORKSPACE_ROOT / "results_local"
MANIFEST_PATH = WORKSPACE_ROOT / "dav_files_manifest.json"
SESSION_PATH = WORKSPACE_ROOT / "session.json"
PROMPT_PATH = WORKSPACE_ROOT / "prompt.txt"
LOCK_PATH = WORKSPACE_ROOT / "session.lock"
PIPELINE_PID_PATH = WORKSPACE_ROOT / "pipeline.pid"
SAM3_REPO = Path("/kaggle/working/sam3")
MINIFORGE_ROOT = Path("/kaggle/working/miniforge3")
ENV_PREFIX = MINIFORGE_ROOT / "envs" / "sam3"
MEGA_ROOT = "/SAM3"
MEGA_RESULTS_ROOT = "/SAM3/results"
PROMPTS = ["vehicle", "person", "animal", "road", "building", "wheel"]
CHUNK_FRAMES = int(os.environ.get("SAM3_CHUNK_FRAMES", "100"))
LOW_DISK_BYTES = 4 * 1024**3
CHECKPOINT_PATH = "/root/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
BPE_PATH = "/kaggle/working/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
HOSTNAME = socket.gethostname()


def iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def worker_pid_path(worker_name: str) -> Path:
    return WORKSPACE_ROOT / f"{worker_name}.pid"


def ensure_dirs():
    for path in [WORKSPACE_ROOT, LOGS_DIR, TMP_DIR, RESULTS_LOCAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def release_cuda_memory():
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if hasattr(torch.cuda, "ipc_collect"):
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_cmd(args, check=True, capture=False, env=None):
    kwargs = {"text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.run(args, env=env, **kwargs)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {args}\n"
            f"stdout={getattr(proc, 'stdout', '')}\n"
            f"stderr={getattr(proc, 'stderr', '')}"
        )
    return proc


def mega_exists(remote_path: str) -> bool:
    return run_cmd(["mega-ls", remote_path], check=False).returncode == 0


def mega_mkdir(remote_path: str):
    run_cmd(["mega-mkdir", "-p", remote_path], check=False)


def mega_rm(remote_path: str):
    run_cmd(["mega-rm", remote_path], check=False)


def mega_upload(local_path: Path, remote_dir: str, remote_name: str | None = None):
    mega_mkdir(remote_dir)
    remote_name = remote_name or local_path.name
    mega_rm(f"{remote_dir.rstrip('/')}/{remote_name}")
    run_cmd(["mega-put", "-c", str(local_path), remote_dir], check=True)


def mega_download(remote_path: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    proc = run_cmd(["mega-get", remote_path, str(local_dir)], check=False, capture=True)
    return proc.returncode == 0


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def base_prompt_state():
    return {
        "status": "pending",
        "completed_chunks": [],
        "failed_chunks": [],
        "uploaded_result_paths": [],
        "chunk_count": None,
        "last_error": None,
        "started_at": None,
        "completed_at": None,
    }


def init_session_from_manifest(manifest):
    items = []
    for file_item in manifest["files"]:
        items.append(
            {
                "manifest_index": file_item["manifest_index"],
                "filename": file_item["filename"],
                "source_url": file_item["source_url"],
                "source_id": file_item["source_file_id"],
                "source_folder_id": file_item["source_folder_id"],
                "status": "pending",
                "claim": None,
                "preprocessing": {
                    "dav_path": None,
                    "mp4_path": None,
                    "chunk_dir": None,
                    "chunk_count": None,
                    "converted_at": None,
                },
                "prompts": {prompt: base_prompt_state() for prompt in PROMPTS},
                "stop_reason": None,
                "errors": [],
            }
        )
    return {
        "version": 1,
        "updated_at": iso_now(),
        "workspace_root": str(WORKSPACE_ROOT),
        "repo_root": str(SAM3_REPO),
        "env_root": str(MINIFORGE_ROOT),
        "mega": {
            "root": MEGA_ROOT,
            "session_path": "/SAM3/session.json",
            "prompt_path": "/SAM3/prompt.txt",
            "authenticated": True,
        },
        "manifest": {
            "path": str(MANIFEST_PATH),
            "source_type": manifest["source_type"],
            "source_ref": manifest["source_ref"],
            "source_url": manifest["source_url"],
            "count": manifest["count"],
            "next_index": 0,
        },
        "prompts": PROMPTS,
        "workers": {
            "worker_a": {"gpu": 0, "pid": None, "status": "idle", "claimed_task": None, "last_heartbeat": None},
            "worker_b": {"gpu": 1, "pid": None, "status": "idle", "claimed_task": None, "last_heartbeat": None},
        },
        "current": {"claimed_tasks": {}},
        "items": items,
        "summary": {},
        "errors": [],
        "stop_reason": None,
        "resume_hint": "Read session.json, keep whole-video claims atomic, and sync every state change back to MEGA.",
        "lock": {"path": str(LOCK_PATH), "owner": None, "held": False},
    }


def merge_existing_session(session, manifest):
    by_index = {item["manifest_index"]: item for item in session.get("items", [])}
    merged = []
    for file_item in manifest["files"]:
        existing = by_index.get(file_item["manifest_index"])
        if existing is None:
            existing = {
                "manifest_index": file_item["manifest_index"],
                "filename": file_item["filename"],
                "source_url": file_item["source_url"],
                "source_id": file_item["source_file_id"],
                "source_folder_id": file_item["source_folder_id"],
                "status": "pending",
                "claim": None,
                "preprocessing": {
                    "dav_path": None,
                    "mp4_path": None,
                    "chunk_dir": None,
                    "chunk_count": None,
                    "converted_at": None,
                },
                "prompts": {},
                "stop_reason": None,
                "errors": [],
            }
        existing["filename"] = file_item["filename"]
        existing["source_url"] = file_item["source_url"]
        existing["source_id"] = file_item["source_file_id"]
        existing["source_folder_id"] = file_item["source_folder_id"]
        existing.setdefault(
            "preprocessing",
            {"dav_path": None, "mp4_path": None, "chunk_dir": None, "chunk_count": None, "converted_at": None},
        )
        existing.setdefault("claim", None)
        existing.setdefault("errors", [])
        existing.setdefault("prompts", {})
        for prompt in PROMPTS:
            existing["prompts"].setdefault(prompt, base_prompt_state())
        merged.append(existing)
    session["items"] = merged
    session["prompts"] = PROMPTS
    session["manifest"].update(
        {
            "path": str(MANIFEST_PATH),
            "source_type": manifest["source_type"],
            "source_ref": manifest["source_ref"],
            "source_url": manifest["source_url"],
            "count": manifest["count"],
        }
    )
    return session


def update_summary(session):
    total_prompts = len(session["items"]) * len(PROMPTS)
    completed_prompts = 0
    failed_prompts = 0
    claimed_tasks = 0
    uploaded_results = 0
    next_index = len(session["items"])
    for item in session["items"]:
        if item.get("claim"):
            claimed_tasks += 1
        if item["status"] != "completed" and next_index == len(session["items"]):
            next_index = item["manifest_index"]
        for prompt in PROMPTS:
            pstate = item["prompts"][prompt]
            if pstate["status"] == "completed":
                completed_prompts += 1
            elif pstate["status"] == "failed":
                failed_prompts += 1
            uploaded_results += len(pstate.get("uploaded_result_paths", []))
    session["manifest"]["next_index"] = next_index if session["items"] else 0
    session["summary"] = {
        "total_manifest_items": len(session["items"]),
        "total_prompts": total_prompts,
        "claimed_tasks": claimed_tasks,
        "completed_prompts": completed_prompts,
        "failed_prompts": failed_prompts,
        "uploaded_results": uploaded_results,
    }
    session["updated_at"] = iso_now()


def build_resume_prompt(session):
    next_item = None
    for item in session["items"]:
        if item["status"] != "completed":
            next_item = item
            break
    next_action = "All known work is complete."
    if next_item is not None:
        next_action = f"Resume with manifest_index={next_item['manifest_index']} filename={next_item['filename']}"
    stop_reason = session.get("stop_reason") or "running"
    return f"""Resume SAM3 from MEGA state only.

State:
- Read `session.json` first.
- Continue from `manifest.next_index` and respect any existing whole-video claims.
- Never duplicate completed prompts or uploaded results.
- Use two single-GPU workers only:
  - worker A -> GPU 0
  - worker B -> GPU 1
- Claim work atomically before processing.

Rules:
- Download DAV from the manifest source file id.
- Convert DAV to 15 fps MP4.
- Split into 200-frame chunks.
- Run prompts: vehicle, person, animal, road, building, wheel.
- Sync every meaningful state change back to MEGA.
- Remove temporary files after successful upload when safe.
- If disk is low or MEGA sync fails, checkpoint and stop cleanly.

Current stop reason: {stop_reason}
Next action: {next_action}
Summary: {json.dumps(session.get('summary', {}), sort_keys=True)}
"""


def save_session(session):
    update_summary(session)
    SESSION_PATH.write_text(json.dumps(session, indent=2), encoding="utf-8")
    PROMPT_PATH.write_text(build_resume_prompt(session), encoding="utf-8")
    mega_mkdir(MEGA_ROOT)
    mega_mkdir(MEGA_RESULTS_ROOT)
    mega_upload(SESSION_PATH, MEGA_ROOT)
    mega_upload(PROMPT_PATH, MEGA_ROOT)


def load_or_init_session():
    ensure_dirs()
    manifest = load_manifest()
    if not SESSION_PATH.exists() and mega_exists("/SAM3/session.json"):
        mega_download("/SAM3/session.json", WORKSPACE_ROOT)
    if SESSION_PATH.exists():
        session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        session = merge_existing_session(session, manifest)
    else:
        session = init_session_from_manifest(manifest)
        save_session(session)
    update_summary(session)
    return session


def get_item(session, manifest_index):
    for item in session["items"]:
        if item["manifest_index"] == manifest_index:
            return item
    raise KeyError(manifest_index)


def disk_low() -> bool:
    return shutil.disk_usage(WORKSPACE_ROOT).free < LOW_DISK_BYTES


def update_worker(session, worker_name, status=None, claimed_task=None, pid=None):
    worker = session["workers"][worker_name]
    if status is not None:
        worker["status"] = status
    worker["claimed_task"] = claimed_task
    if pid is not None:
        worker["pid"] = pid
    worker["last_heartbeat"] = iso_now()


def lock_handle(handle):
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


def unlock_handle(handle):
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def session_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+", encoding="utf-8")
    if handle.tell() == 0:
        handle.write("\n")
        handle.flush()
    handle.seek(0)
    lock_handle(handle)
    return handle


def claim_next_item(worker_name):
    handle = session_lock()
    try:
        session = load_or_init_session()
        if session.get("stop_reason") == "low_disk":
            return None, session
        existing = session["workers"][worker_name].get("claimed_task")
        if existing is not None:
            return get_item(session, existing["manifest_index"]), session
        for item in session["items"]:
            if item["status"] in {"completed", "failed"}:
                continue
            if item.get("claim") is not None:
                continue
            item["claim"] = {"worker": worker_name, "claimed_at": iso_now(), "host": HOSTNAME}
            item["status"] = "claimed"
            update_worker(
                session,
                worker_name,
                status="busy",
                claimed_task={"manifest_index": item["manifest_index"], "filename": item["filename"]},
                pid=os.getpid(),
            )
            session["current"]["claimed_tasks"][worker_name] = {
                "manifest_index": item["manifest_index"],
                "filename": item["filename"],
                "claimed_at": iso_now(),
            }
            save_session(session)
            return item, session
        update_worker(session, worker_name, status="idle", claimed_task=None, pid=os.getpid())
        session["current"]["claimed_tasks"].pop(worker_name, None)
        save_session(session)
        return None, session
    finally:
        unlock_handle(handle)
        handle.close()


def heartbeat(worker_name):
    handle = session_lock()
    try:
        session = load_or_init_session()
        worker = session["workers"][worker_name]
        update_worker(
            session,
            worker_name,
            status=worker.get("status", "idle"),
            claimed_task=worker.get("claimed_task"),
            pid=os.getpid(),
        )
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def release_item(worker_name, manifest_index, final_status=None, stop_reason=None, error_text=None):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        if error_text:
            item.setdefault("errors", []).append({"at": iso_now(), "message": error_text})
        item["claim"] = None
        if stop_reason:
            item["stop_reason"] = stop_reason
            session["stop_reason"] = stop_reason
        if final_status:
            item["status"] = final_status
        update_worker(session, worker_name, status="idle", claimed_task=None, pid=os.getpid())
        session["current"]["claimed_tasks"].pop(worker_name, None)
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def note_preprocessing(manifest_index, dav_path, mp4_path, chunk_dir, chunk_count):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        item["preprocessing"].update(
            {
                "dav_path": str(dav_path),
                "mp4_path": str(mp4_path),
                "chunk_dir": str(chunk_dir),
                "chunk_count": chunk_count,
                "converted_at": iso_now(),
            }
        )
        for prompt in PROMPTS:
            item["prompts"][prompt]["chunk_count"] = chunk_count
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def note_prompt_start(manifest_index, prompt):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        pstate = item["prompts"][prompt]
        if pstate["started_at"] is None:
            pstate["started_at"] = iso_now()
        pstate["status"] = "in_progress"
        pstate["last_error"] = None
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def note_chunk_upload(manifest_index, prompt, chunk_index, remote_path):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        pstate = item["prompts"][prompt]
        if chunk_index not in pstate["completed_chunks"]:
            pstate["completed_chunks"].append(chunk_index)
            pstate["completed_chunks"].sort()
        if remote_path not in pstate["uploaded_result_paths"]:
            pstate["uploaded_result_paths"].append(remote_path)
        if chunk_index in pstate["failed_chunks"]:
            pstate["failed_chunks"].remove(chunk_index)
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def note_prompt_complete(manifest_index, prompt):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        pstate = item["prompts"][prompt]
        pstate["status"] = "completed"
        pstate["completed_at"] = iso_now()
        if all(item["prompts"][p]["status"] == "completed" for p in PROMPTS):
            item["status"] = "completed"
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def note_prompt_failure(manifest_index, prompt, chunk_index, message):
    handle = session_lock()
    try:
        session = load_or_init_session()
        item = get_item(session, manifest_index)
        pstate = item["prompts"][prompt]
        pstate["status"] = "failed"
        pstate["last_error"] = message
        if chunk_index is not None and chunk_index not in pstate["failed_chunks"]:
            pstate["failed_chunks"].append(chunk_index)
            pstate["failed_chunks"].sort()
        item["status"] = "failed"
        item.setdefault("errors", []).append({"at": iso_now(), "message": f"{prompt}: {message}"})
        save_session(session)
    finally:
        unlock_handle(handle)
        handle.close()


def ensure_mega_layout():
    mega_mkdir("/SAM3")
    mega_mkdir("/SAM3/results")


def ensure_local_video_assets(item):
    video_key = safe_name(item["filename"])
    video_dir = TMP_DIR / video_key
    chunk_dir = video_dir / "chunks"
    video_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    dav_path = video_dir / item["filename"]
    mp4_path = video_dir / f"{video_key}_15fps.mp4"
    if not dav_path.exists():
        run_cmd(["gdown", f"https://drive.google.com/uc?id={item['source_id']}", "-O", str(dav_path)])
    if not mp4_path.exists():
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(dav_path),
                "-an",
                "-vf",
                "fps=15",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-g",
                str(CHUNK_FRAMES),
                "-keyint_min",
                str(CHUNK_FRAMES),
                "-sc_threshold",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ]
        )
    chunk_files = sorted(chunk_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_path),
                "-c",
                "copy",
                "-map",
                "0:v:0",
                "-f",
                "segment",
                "-segment_frames",
                str(CHUNK_FRAMES),
                "-reset_timestamps",
                "1",
                str(chunk_dir / "chunk_%04d.mp4"),
            ]
        )
        chunk_files = sorted(chunk_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        raise RuntimeError(f"no chunks created for {mp4_path}")
    note_preprocessing(item["manifest_index"], dav_path, mp4_path, chunk_dir, len(chunk_files))
    return dav_path, mp4_path, chunk_dir, chunk_files


def make_predictor():
    return Sam3VideoPredictor(compile=False, checkpoint_path=CHECKPOINT_PATH, bpe_path=BPE_PATH)


def encode_masks(masks):
    encoded = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        encoded.append({"size": rle["size"], "counts": rle["counts"].decode("ascii")})
    return encoded


def run_chunk(predictor, chunk_path: Path, prompt: str):
    response = predictor.handle_request({"type": "start_session", "resource_path": str(chunk_path)})
    session_id = response["session_id"]
    try:
        predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt})
        frames = []
        for item in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": CHUNK_FRAMES,
            }
        ):
            outputs = item["outputs"]
            frames.append(
                {
                    "frame_index": int(item["frame_index"]),
                    "obj_ids": outputs["out_obj_ids"].tolist(),
                    "probs": np.asarray(outputs["out_probs"]).tolist(),
                    "boxes_xywh": np.asarray(outputs["out_boxes_xywh"]).tolist(),
                    "mask_rles": encode_masks(np.asarray(outputs["out_binary_masks"])),
                    "frame_stats": outputs.get("frame_stats"),
                }
            )
        return frames
    finally:
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        finally:
            release_cuda_memory()


def write_chunk_result(item, prompt, chunk_index, chunk_file, frames):
    video_key = safe_name(item["filename"])
    out_dir = RESULTS_LOCAL_DIR / video_key / prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{chunk_index:04d}.json.gz"
    payload = {
        "manifest_index": item["manifest_index"],
        "filename": item["filename"],
        "source_id": item["source_id"],
        "prompt": prompt,
        "chunk_index": chunk_index,
        "chunk_file": chunk_file.name,
        "created_at": iso_now(),
        "frames": frames,
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    remote_dir = f"{MEGA_RESULTS_ROOT}/{video_key}/{prompt}"
    mega_upload(out_path, remote_dir)
    return f"{remote_dir}/chunk_{chunk_index:04d}.json.gz"


def cleanup_video_assets(dav_path, mp4_path, chunk_dir):
    for path in [dav_path, mp4_path]:
        path = Path(path)
        if path.exists():
            path.unlink()
    if Path(chunk_dir).exists():
        shutil.rmtree(chunk_dir, ignore_errors=True)


def process_item(worker_name, item):
    if disk_low():
        release_item(worker_name, item["manifest_index"], final_status=item["status"], stop_reason="low_disk")
        return False
    dav_path, mp4_path, chunk_dir, chunk_files = ensure_local_video_assets(item)
    predictor = make_predictor()
    try:
        for prompt in PROMPTS:
            pstate = item["prompts"][prompt]
            if pstate["status"] == "completed":
                continue
            note_prompt_start(item["manifest_index"], prompt)
            for chunk_index, chunk_file in enumerate(chunk_files):
                heartbeat(worker_name)
                if chunk_index in pstate["completed_chunks"]:
                    continue
                if disk_low():
                    release_item(worker_name, item["manifest_index"], final_status="pending", stop_reason="low_disk")
                    return False
                try:
                    frames = run_chunk(predictor, chunk_file, prompt)
                    remote_path = write_chunk_result(item, prompt, chunk_index, chunk_file, frames)
                    note_chunk_upload(item["manifest_index"], prompt, chunk_index, remote_path)
                except Exception as exc:
                    note_prompt_failure(
                        item["manifest_index"],
                        prompt,
                        chunk_index,
                        "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                    )
                    raise
                finally:
                    release_cuda_memory()
            note_prompt_complete(item["manifest_index"], prompt)
        release_item(worker_name, item["manifest_index"], final_status="completed")
        cleanup_video_assets(dav_path, mp4_path, chunk_dir)
        return True
    finally:
        del predictor
        release_cuda_memory()


def worker_loop(worker_name, gpu):
    ensure_dirs()
    ensure_mega_layout()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    while True:
        item, _session = claim_next_item(worker_name)
        if item is None:
            break
        try:
            ok = process_item(worker_name, item)
            if not ok:
                break
        except Exception as exc:
            release_item(
                worker_name,
                item["manifest_index"],
                final_status="failed",
                error_text="".join(traceback.format_exception(exc)).strip(),
            )
            release_cuda_memory()
            continue


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_worker_subprocess(worker_name, gpu):
    log_path = LOGS_DIR / f"{worker_name}.log"
    pid_path = worker_pid_path(worker_name)
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text().strip())
        except Exception:
            existing = None
        if existing and pid_alive(existing):
            return existing
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        str(ENV_PREFIX / "bin" / "python"),
        str(WORKSPACE_ROOT / "sam3_remote_pipeline.py"),
        "worker",
        "--worker-name",
        worker_name,
        "--gpu",
        str(gpu),
    ]
    with log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(WORKSPACE_ROOT),
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def launch():
    ensure_dirs()
    ensure_mega_layout()
    session = load_or_init_session()
    save_session(session)
    PIPELINE_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    pid_a = start_worker_subprocess("worker_a", 0)
    pid_b = start_worker_subprocess("worker_b", 1)
    print(json.dumps({"worker_a": pid_a, "worker_b": pid_b}, indent=2))


def status():
    data = {}
    for worker_name in ["worker_a", "worker_b"]:
        pid_path = worker_pid_path(worker_name)
        pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
        data[worker_name] = {"pid": pid, "alive": pid_alive(pid)}
    if SESSION_PATH.exists():
        data["session"] = json.loads(SESSION_PATH.read_text(encoding="utf-8")).get("summary", {})
    print(json.dumps(data, indent=2))


def stop():
    for worker_name in ["worker_a", "worker_b"]:
        pid_path = worker_pid_path(worker_name)
        if not pid_path.exists():
            continue
        pid = int(pid_path.read_text().strip())
        try:
            os.killpg(pid, 15)
        except Exception:
            try:
                os.kill(pid, 15)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_worker = sub.add_parser("worker")
    p_worker.add_argument("--worker-name", required=True)
    p_worker.add_argument("--gpu", required=True, type=int)
    sub.add_parser("launch")
    sub.add_parser("status")
    sub.add_parser("stop")
    args = parser.parse_args()
    if args.cmd == "worker":
        worker_loop(args.worker_name, args.gpu)
    elif args.cmd == "launch":
        launch()
    elif args.cmd == "status":
        status()
    elif args.cmd == "stop":
        stop()


if __name__ == "__main__":
    main()
