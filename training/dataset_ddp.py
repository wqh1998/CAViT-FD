from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import numpy as np
import random
import re, os
from torchvision import transforms
import torch

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.path = path
        self.resolution = resolution
        self.transform = transform
        self.length = None

    def _open(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError(f"Cannot open lmdb dataset {self.path}")

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def _close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __len__(self):
        if self.length is None:
            self._open()
            self._close()

        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._open()

        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
