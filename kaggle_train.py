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

restored = False

# Search all known checkpoint locations under /kaggle/input
_search_patterns = [
    "**/dsbm_checkpoints/*.ckpt",       # Kaggle auto-extracted zip
    "**/dsbm_run/checkpoints/*.ckpt",   # direct output (Cell 6 didn't run)
    "**/checkpoints/*.ckpt",            # any checkpoints folder
]
_ckpt_files = []
for _pat in _search_patterns:
    _ckpt_files = _glob.glob(f"/kaggle/input/{_pat}", recursive=True)
    if _ckpt_files:
        break

if _ckpt_files:
    for f in _ckpt_files:
        shutil.copy(f, CKPT_DIR)
    print(f"Restored {len(_ckpt_files)} checkpoint files from {os.path.dirname(_ckpt_files[0])}.")
    restored = True

# Fallback: zip file
if not restored:
    prev_zips = _glob.glob("/kaggle/input/**/dsbm_checkpoints.zip", recursive=True)
    if prev_zips:
        shutil.unpack_archive(prev_zips[0], CKPT_DIR)
        print(f"Restored {len(os.listdir(CKPT_DIR))} checkpoint files from zip.")
        restored = True

if not restored:
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

_PREFIXES = ["net_b", "sample_net_b", "optimizer_b", "net_f", "sample_net_f", "optimizer_f"]

# For each prefix, find the latest checkpoint file (by filename sort = chronological)
_latest = {}
for _p in _PREFIXES:
    _matches = sorted(_glob.glob(f"{CKPT_DIR}/{_p}_*.ckpt"))
    if _matches:
        _latest[_p] = _matches[-1]

# Only include a direction if all three files of that direction exist
_b = [_latest.get(p) for p in ["net_b", "sample_net_b", "optimizer_b"]]
_f = [_latest.get(p) for p in ["net_f", "sample_net_f", "optimizer_f"]]

files_to_zip = [f for f in _b if f is not None]
if all(f is not None for f in _f):
    files_to_zip.extend(_f)

print(f"Saving {len(files_to_zip)} checkpoint files (latest set only) ...")
with zipfile.ZipFile(CKPT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_zip:
        zf.write(f, os.path.basename(f))

size_mb = os.path.getsize(CKPT_ZIP) / 1e6
print(f"Saved {CKPT_ZIP}  ({size_mb:.0f} MB)")
print("→ Output tab → 'New Dataset' → name it 'dsbm-checkpoints'")


# ── CELL 7: View sample outputs ───────────────────────────────────────────────
import glob as _glob
from PIL import Image as PILImage

images = sorted(_glob.glob(f"{RUN_DIR}/im/**/*.png", recursive=True))
print(f"{len(images)} sample images generated.")
if images:
    print("Latest:", images[-1])
    display(PILImage.open(images[-1]))
