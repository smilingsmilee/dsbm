from pathlib import Path
import torch
from PIL import Image
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(torch.utils.data.Dataset):
    """Loads all images from a single flat folder. Works with any image format."""

    def __init__(self, folder, image_size=None):
        p = Path(folder)
        assert p.is_dir(), f"Not a directory: {p}"
        self.paths = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        assert len(self.paths) > 0, f"No images found in {p}"
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0 * 2 - 1
        return t.permute(2, 0, 1), torch.zeros((1,))
