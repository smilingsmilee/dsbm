# ============================================================
# DSBM Unpaired Image Translation  —  Kaggle Notebook
# ============================================================
# Paste each CELL block into a separate Kaggle notebook cell.
#
# One-time setup:
#   1. New Notebook → Settings → Accelerator → GPU T4 x1
#   2. Settings → Internet → On
#   3. Add Data → your image dataset (two domain folders)
#   4. (Optional) Add Data → "dsbm-checkpoints" to resume a previous session
# ============================================================


# ── CELL 1: Install dependencies ──────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "hydra-core==1.3.2",
    "omegaconf",
    "POT",
    "torchdiffeq",
    "accelerate",
    "torch-fidelity",
], check=True)
print("Dependencies installed.")


# ── CELL 2: Clone codebase and verify dataset ─────────────────────────────────
import os, shutil, glob as _glob

GITHUB_REPO = "https://github.com/smilingsmilee/dsbm.git"
CLONE_DST   = "/kaggle/working/dsbm"
RUN_DIR     = "/kaggle/working/dsbm_run"
CKPT_DIR    = f"{RUN_DIR}/checkpoints"
CKPT_ZIP    = "/kaggle/working/dsbm_checkpoints.zip"

# ── Set your two domain folders here ──────────────────────────────────────────
DOMAIN_A = "/kaggle/input/YOUR_DATASET/domain_a"   # ← change this
DOMAIN_B = "/kaggle/input/YOUR_DATASET/domain_b"   # ← change this
# ──────────────────────────────────────────────────────────────────────────────

if os.path.exists(CLONE_DST):
    shutil.rmtree(CLONE_DST)
ret = os.system(f"git clone {GITHUB_REPO} {CLONE_DST}")
assert ret == 0, "git clone failed — check internet access is enabled."

os.chdir(CLONE_DST)
print("Working directory:", os.getcwd())

a_imgs = _glob.glob(f"{DOMAIN_A}/*")
b_imgs = _glob.glob(f"{DOMAIN_B}/*")
print(f"Domain A: {len(a_imgs)} images  ({DOMAIN_A})")
print(f"Domain B: {len(b_imgs)} images  ({DOMAIN_B})")
assert len(a_imgs) > 0, f"No images found in DOMAIN_A: {DOMAIN_A}"
assert len(b_imgs) > 0, f"No images found in DOMAIN_B: {DOMAIN_B}"


# ── CELL 3: Verify GPU ────────────────────────────────────────────────────────
import torch
assert torch.cuda.is_available(), "No GPU found — enable GPU in Settings."
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── CELL 4: Restore checkpoints from a previous session (if any) ──────────────
import glob as _glob, shutil, os

os.makedirs(CKPT_DIR, exist_ok=True)

prev_zips = _glob.glob("/kaggle/input/*/dsbm_checkpoints.zip")
if prev_zips:
    src_zip = prev_zips[0]
    print(f"Found previous checkpoints: {src_zip}")
    shutil.unpack_archive(src_zip, CKPT_DIR)
    print(f"Restored {len(os.listdir(CKPT_DIR))} checkpoint files.")
else:
    print("No previous checkpoints found — starting from scratch.")


# ── CELL 5: Train ─────────────────────────────────────────────────────────────
# Resumes automatically if checkpoints exist in CKPT_DIR.
# To switch to DSBM-IMF add:  first_coupling=ind first_num_iter=10000
# To train longer add:        n_ipf=20 num_iter=10000

import os
cmd = (
    f"python main.py "
    f"dataset=custom_transfer "
    f"LOGGER=CSV "
    f"method=dbdsb "
    f"model=UNET "
    f"++paths.init_data_path={DOMAIN_A} "
    f"++paths.final_data_path={DOMAIN_B} "
    f"'hydra.run.dir={RUN_DIR}'"
)

print("Running:", cmd)
os.system(cmd)


# ── CELL 6: Save checkpoints for the next session ─────────────────────────────
# After this cell runs:
#   1. Output tab → "New Dataset" → name it "dsbm-checkpoints"
#   2. In your next session, add that dataset — Cell 4 restores it automatically.

import glob as _glob, zipfile, os

ckpt_files = _glob.glob(f"{CKPT_DIR}/*.ckpt")
print(f"Saving {len(ckpt_files)} checkpoint files ...")

with zipfile.ZipFile(CKPT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in ckpt_files:
        zf.write(f, os.path.basename(f))

print(f"Saved {CKPT_ZIP}  ({os.path.getsize(CKPT_ZIP) / 1e6:.0f} MB)")
print("→ Output tab → 'New Dataset' → name it 'dsbm-checkpoints'")


# ── CELL 7: View sample outputs ───────────────────────────────────────────────
import glob as _glob
from PIL import Image as PILImage

images = sorted(_glob.glob(f"{RUN_DIR}/im/**/*.png", recursive=True))
print(f"{len(images)} sample images generated.")
if images:
    print("Latest:", images[-1])
    display(PILImage.open(images[-1]))
