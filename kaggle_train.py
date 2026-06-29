# ============================================================
# DSBM Cat→Dog Image Translation  —  Kaggle Notebook
# ============================================================
# Paste each section into a separate Kaggle notebook cell.
#
# One-time setup before first run:
#   1. Create a new Kaggle Notebook
#   2. Settings → Accelerator → GPU T4 x1
#   3. Add dataset: search "animal faces" → add "Animal Faces HQ (AFHQ)"
#      by andrewmvd  (slug: andrewmvd/animal-faces)
#   4. Enable internet access: Settings → Internet → On
#
# To resume a previous session:
#   5. In the previous session's notebook, click Output tab → "New Dataset"
#      and name it "dsbm-checkpoints" — this saves /kaggle/working/ as a dataset
#   6. In the new notebook, add that "dsbm-checkpoints" dataset
#      The restore cell below will detect and load it automatically.
# ============================================================

GITHUB_REPO = "https://github.com/smilingsmilee/dsbm.git"

# Checkpoints are always written to this fixed path so they are easy to find
# across sessions.
RUN_DIR   = "/kaggle/working/dsbm_run"
CKPT_DIR  = f"{RUN_DIR}/checkpoints"
CKPT_ZIP  = "/kaggle/working/dsbm_checkpoints.zip"   # saved at end of each session


# ── CELL 1: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "hydra-core==1.3.2",
    "omegaconf",
    "POT",
    "torchdiffeq",
    "accelerate",
], check=True)
print("Dependencies installed.")


# ── CELL 2: Clone codebase from GitHub ────────────────────────────────────────
import os, shutil

CLONE_DST = "/kaggle/working/dsbm"

if os.path.exists(CLONE_DST):
    shutil.rmtree(CLONE_DST)

ret = os.system(f"git clone {GITHUB_REPO} {CLONE_DST}")
assert ret == 0, "git clone failed — check the repo URL and internet access."

os.chdir(CLONE_DST)
print("Working directory:", os.getcwd())

# Sanity-check AFHQ dataset
import glob as _glob
cats = _glob.glob("/kaggle/input/animal-faces/afhq/train/cat/*")
dogs = _glob.glob("/kaggle/input/animal-faces/afhq/train/dog/*")
print(f"Cat images: {len(cats)},  Dog images: {len(dogs)}")
assert len(cats) > 0 and len(dogs) > 0, \
    "AFHQ dataset not found. Add 'andrewmvd/animal-faces' to this notebook."


# ── CELL 3: Verify GPU ────────────────────────────────────────────────────────
import torch
assert torch.cuda.is_available(), "No GPU found — enable GPU in Settings."
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── CELL 4: Restore checkpoints from a previous session (if any) ──────────────
import glob as _glob, shutil, os

os.makedirs(CKPT_DIR, exist_ok=True)

# Look for a checkpoint zip saved by a previous session and added as a dataset
prev_zips = _glob.glob("/kaggle/input/*/dsbm_checkpoints.zip")

if prev_zips:
    src_zip = prev_zips[0]
    print(f"Found previous checkpoints: {src_zip}")
    shutil.unpack_archive(src_zip, CKPT_DIR)
    ckpts = os.listdir(CKPT_DIR)
    print(f"Restored {len(ckpts)} checkpoint files.")
else:
    print("No previous checkpoints found — starting from scratch.")


# ── CELL 5: Train ─────────────────────────────────────────────────────────────
# hydra.run.dir pins the output to a fixed path so find_last_ckpt() can
# always locate the most recent valid checkpoint on resume.
#
# The trainer automatically detects any checkpoints in CKPT_DIR and resumes
# from the latest consistent (forward, backward) pair — no manual flags needed.
#
# To switch to DSBM-IMF add:  first_coupling=ind first_num_iter=10000
# To train longer add:        n_ipf=20 num_iter=10000

cmd = (
    f"python main.py "
    f"dataset=afhq_cat2dog "
    f"LOGGER=CSV "
    f"method=dbdsb "
    f"model=UNET "
    f"++paths.afhq_path=/kaggle/input/animal-faces/afhq "
    f"'hydra.run.dir={RUN_DIR}'"
)
print("Running:", cmd)
os.system(cmd)


# ── CELL 6: Save checkpoints for the next session ─────────────────────────────
# This zips only the checkpoint files (not images or cache) and writes the zip
# to /kaggle/working/ so it appears in the Output tab.
#
# After this cell runs:
#   1. Click the Output tab in this notebook
#   2. Click "New Dataset" and name it "dsbm-checkpoints"
#   3. In your next session, add that dataset — Cell 4 will auto-restore it.

import glob as _glob, zipfile, os

ckpt_files = _glob.glob(f"{CKPT_DIR}/*.ckpt")
print(f"Saving {len(ckpt_files)} checkpoint files to {CKPT_ZIP} ...")

with zipfile.ZipFile(CKPT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in ckpt_files:
        zf.write(f, os.path.basename(f))

size_mb = os.path.getsize(CKPT_ZIP) / 1e6
print(f"Saved {CKPT_ZIP}  ({size_mb:.0f} MB)")
print("Now go to the Output tab → 'New Dataset' → name it 'dsbm-checkpoints'.")


# ── CELL 7: Inspect sample outputs ────────────────────────────────────────────
import glob as _glob
from PIL import Image as PILImage

images = sorted(_glob.glob(f"{RUN_DIR}/im/**/*.png", recursive=True))
print(f"{len(images)} sample images generated.")

if images:
    print("Latest:", images[-1])
    display(PILImage.open(images[-1]))
