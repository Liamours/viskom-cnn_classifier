import pandas as pd
import torch
from pathlib import Path
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torch.utils.data import Dataset


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
NUM_CLASSES = len(CLASS_NAMES)


def get_transforms(input_size: int) -> v2.Compose:
    return v2.Compose([
        v2.Resize((input_size, input_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])


class BloodMNISTDataset(Dataset):
    def __init__(self, split: str, processed_dir: str, input_size: int = 224):
        self.root      = Path(processed_dir) / split
        self.transform = get_transforms(input_size)

        df = pd.read_csv(Path(processed_dir) / f"{split}.csv")
        self.filenames = df["filename"].tolist()
        self.labels    = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img = read_image(
            str(self.root / self.filenames[idx]),
            mode=ImageReadMode.RGB,
        )
        img = self.transform(img)
        return img, self.labels[idx]
