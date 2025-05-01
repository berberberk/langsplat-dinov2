import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import logging

logger = logging.getLogger("dataset")

def load_npz(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.load(f)["arr_0"]

class AutoencoderDataset(Dataset):
    def __init__(self, data_dir):
        logger.info(f"data_dir: {data_dir}")
        self.data = []
        self.data_dic = {}
        for path in Path(data_dir).glob("*.npz"):
            mat = load_npz(path)
            mat = mat.reshape(mat.shape[0], -1).transpose((1, 0))
            self.data.append(mat)
            self.data_dic[str(path)] = mat
        
        self.data = np.concatenate(self.data)
        logger.info(f"data shape: {self.data.shape}")

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0]
