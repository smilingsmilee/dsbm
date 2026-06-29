import torch
from pathlib import Path
from PIL import Image
import numpy as np


class AFHQ(torch.utils.data.Dataset):
    """
        root_dir is to train set that has the cat, dog, wild folders in
        animal_type is either cat, dog, or wild
        image_size resizes the image if not None (default keeps original 512x512)
    """
    def __init__(self, root_dir, animal_type, image_size=None):
        self.root_dir = root_dir
        self.animal_type = animal_type
        self.image_size = image_size
        assert animal_type in ['cat', 'dog', 'wild']
        p = Path(self.root_dir).joinpath(animal_type)
        self.all_image_paths = sorted(
            list(p.glob('*.png')) + list(p.glob('*.jpg')) + list(p.glob('*.jpeg'))
        )
        assert len(self.all_image_paths) > 0, f"No images found in {p}"

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        path = self.all_image_paths[index]

        pil_image = Image.open(path).convert('RGB')
        if self.image_size is not None:
            pil_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        np_image = np.array(pil_image)

        # scale floats between -1 and 1
        tensor_image = (torch.tensor(np_image, dtype=torch.float32) / 255.0) * 2 - 1

        # (H, W, C) -> (C, H, W)
        tensor_image = tensor_image.permute(2, 0, 1)

        return tensor_image, torch.zeros((1,))





