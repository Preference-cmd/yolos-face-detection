from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
import torch

class FaceData(Dataset):
    def __init__(self, processor, transform = None) -> None:
        super().__init__()
        self.processor = processor
        self.transform = transform

        self.image = Image.open(self.image_path)







    def __len__(self) -> int:
        return len(self.image)


    def __getitem__(self, index: int) -> torch.Tensor:
        pass