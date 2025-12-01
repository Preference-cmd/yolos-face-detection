from torch.utils.data import Dataset
import torch

class FaceData(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        pass


    def __getitem__(self, index: int) -> torch.Tensor:
        pass