from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class HandwritingDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], img_size: int, train: bool = True):
        """
        items: [{"path": "...", "label": int}, ...]
        """
        self.items = items
        self.img_size = img_size
        self.train = train

        if train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomRotation(3),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = Path(item["path"])
        label = item["label"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label, str(img_path)
