"""
Chargement des données FFHQ.

Supporte deux formats :
  - Dossier d'images .jpg/.png  (téléchargement local)
  - Format Arrow (Hugging Face datasets)
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FFHQDataset(Dataset):
    """
    Charge les images FFHQ depuis un dossier local.

    Args:
        folder     : chemin vers le dossier contenant les images
        resolution : taille de redimensionnement (défaut 64)
    """
    def __init__(self, folder, resolution=64):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"Aucune image trouvée dans {folder}")

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


class FFHQArrowDataset(Dataset):
    """
    Charge les images FFHQ depuis le cache Hugging Face (format Arrow).
    Plus rapide que FFHQDataset car pas de conversion en JPG.

    Args:
        split      : portion du dataset ex: 'train', 'train[:100]'
        resolution : taille de redimensionnement (défaut 64)
    """
    def __init__(self, split='train', resolution=64):
        from datasets import load_dataset
        self.ds = load_dataset('bitmind/ffhq-256', split=split)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]['image'].convert('RGB')
        return self.transform(img)
