# Remote SAM3 Automation

This document explains how to use `automate_sam3_remote.py`.

## What It Does

The script automates the remote SAM3 notebook workflow from your local machine over SSH.

It can:

- verify the remote connection
- verify MEGA login state
- create the remote workspace
- clone and prepare the `sam3` repo
- install and update the remote Python environment
- patch SAM3 for T4-safe autocast behavior
- authenticate Hugging Face and cache the checkpoint
- generate the DAV manifest from the Google Drive folder
- upload the remote worker pipeline files
- launch the 2-worker GPU pipeline
- show pipeline status
- show a live `samtop` terminal dashboard with queue and worker progress

## Files Used

The script depends on these local files:

- `automate_sam3_remote.py`
- `sam3_remote_pipeline.py`
- `run_pipeline.sh`

These are uploaded to the remote notebook when needed.

For local terminal monitoring, this repo also includes:

- `samtop.bat` for Windows
- `samtop.sh` for Ubuntu/Linux

The remote pipeline now defaults to `100` frames per chunk to reduce GPU memory pressure. You can override that with `SAM3_CHUNK_FRAMES`.

## Requirements

Local machine:

- Python 3.10+
- `paramiko` installed

Remote notebook:

- reachable over SSH
- MEGA already logged in
- outbound internet access for GitHub, Hugging Face, and Google Drive

## Default Remote Target

By default the script connects to:

- host: `127.0.0.1`
- port: `10022`
- username: `notebook`

It also includes defaults for:

- Hugging Face token
- remote workspace paths

You must provide these explicitly:

- `--password`
- `--drive-folder-id`
- `--drive-folder-url`

You can override the rest with CLI flags.

You can also provide these through:

- `--env-file`
- `--config-file`

## Commands

### 1. Verify

Checks remote connectivity, Python, working directory, GPU visibility, and MEGA auth.

```bash
python automate_sam3_remote.py verify --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 2. Setup

Runs the remote preparation steps without launching workers.

```bash
python automate_sam3_remote.py setup --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 3. Upload Pipeline Only

Uploads the latest local remote-worker scripts to the notebook.

```bash
python automate_sam3_remote.py upload-pipeline --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 4. Launch

Starts the 2-worker remote pipeline.

```bash
python automate_sam3_remote.py launch --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 5. Status

Prints worker state, `nvidia-smi`, and session summary.

```bash
python automate_sam3_remote.py status --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 6. Full

Runs setup, uploads the pipeline, launches it, and prints status.

```bash
python automate_sam3_remote.py full --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```

### 7. Samtop

Shows a live terminal dashboard similar to `htop` for the remote SAM3 pipeline.

It displays:

- overall video and prompt completion bars
- pending, working, and failed counts
- the current file claimed by each worker
- per-worker prompt/chunk progress
- GPU utilization and VRAM usage

Run it directly through Python:

```bash
python automate_sam3_remote.py samtop --password YOUR_SSH_PASSWORD
```

Or from this repo on Ubuntu/Linux:

```bash
chmod +x samtop.sh
./samtop.sh
```

Or on Windows in PowerShell / `cmd`:

```bash
samtop
```

Helpful options:

```bash
python automate_sam3_remote.py samtop --refresh-seconds 1
python automate_sam3_remote.py samtop --once
```

Press `q` to quit the live view.

## .env Mode

You can store settings in a `.env`-style file and pass it with `--env-file`.

Example `.env`:

```dotenv
SAM3_HOST=127.0.0.1
SAM3_PORT=10022
SAM3_USERNAME=notebook
SAM3_PASSWORD=YOUR_SSH_PASSWORD
SAM3_HF_TOKEN=YOUR_HF_TOKEN
SAM3_DRIVE_FOLDER_ID=YOUR_FOLDER_ID
SAM3_DRIVE_FOLDER_URL=YOUR_FOLDER_URL
SAM3_REMOTE_WORKSPACE=/kaggle/working/SAM3
SAM3_REMOTE_REPO=/kaggle/working/sam3
SAM3_REMOTE_MINIFORGE=/kaggle/working/miniforge3
```

Run with:

```bash
python automate_sam3_remote.py full --env-file .env.sam3
```

If `--env-file` is omitted, the script now auto-loads `./.env` when present.

## JSON Config Mode

You can also use a JSON config file.

Example `sam3_remote.json`:

```json
{
  "host": "127.0.0.1",
  "port": 10022,
  "username": "notebook",
  "password": "YOUR_SSH_PASSWORD",
  "hf_token": "YOUR_HF_TOKEN",
  "drive_folder_id": "YOUR_FOLDER_ID",
  "drive_folder_url": "YOUR_FOLDER_URL",
  "remote_workspace": "/kaggle/working/SAM3",
  "remote_repo": "/kaggle/working/sam3",
  "remote_miniforge": "/kaggle/working/miniforge3"
}
```

Run with:

```bash
python automate_sam3_remote.py full --config-file sam3_remote.json
```

## Precedence

Settings are resolved in this order:

1. script defaults
2. `.env` file
3. JSON config file
4. CLI flags

That means CLI flags win if the same value appears in multiple places.

## Useful Overrides

Examples:

```bash
python automate_sam3_remote.py full --host 127.0.0.1 --port 10022 --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
python automate_sam3_remote.py setup --hf-token YOUR_TOKEN
python automate_sam3_remote.py setup --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
python automate_sam3_remote.py status --password YOUR_SSH_PASSWORD
python automate_sam3_remote.py full --env-file .env.sam3
python automate_sam3_remote.py full --config-file sam3_remote.json
```

## Remote Paths

The script assumes this remote layout:

- workspace: `/kaggle/working/SAM3`
- repo: `/kaggle/working/sam3`
- miniforge: `/kaggle/working/miniforge3`

Primary remote outputs:

- `/kaggle/working/SAM3/dav_files_manifest.json`
- `/kaggle/working/SAM3/session.json`
- `/kaggle/working/SAM3/prompt.txt`
- `/kaggle/working/SAM3/logs/`

MEGA outputs:

- `/SAM3/session.json`
- `/SAM3/prompt.txt`
- `/SAM3/results/`

## Notes

- `setup` and `full` expect MEGA to already be logged in remotely.
- The script uses the single-GPU production worker path, not the repo’s distributed multi-GPU predictor.
- The repo is patched remotely to avoid hard-coded `bfloat16` behavior on Tesla T4 GPUs.
- `status` is read-only and safe to run while the pipeline is active.

## Troubleshooting

If `verify` fails:

- check that the local proxy is running on `127.0.0.1:10022`
- confirm the notebook-side SSH server is still alive

If MEGA fails:

- log in on the remote notebook first with `mega-login`
- rerun `python automate_sam3_remote.py verify`

If the pipeline scripts change locally:

- rerun `python automate_sam3_remote.py upload-pipeline`

If you want a fresh end-to-end redeploy:

```bash
python automate_sam3_remote.py full --password YOUR_SSH_PASSWORD --drive-folder-id YOUR_FOLDER_ID --drive-folder-url YOUR_FOLDER_URL
```
