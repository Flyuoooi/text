import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# def read_image(img_path):
#     """Keep reading image until succeed."""
#     if not osp.exists(img_path):
#         raise IOError("{} does not exist".format(img_path))
#     while True:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             return img
#         except IOError:
#             print(f"IOError incurred when reading '{img_path}'. Retrying...")
#             continue
from torchvision.io import read_image as tv_read_image
from torchvision.transforms.functional import to_pil_image

# def read_image(img_path):
#     """高速读取，自动兼容 PIL 和 Tensor."""
#     try:
#         # C++ 后端读取，tensor: [C, H, W], uint8
#         img = tv_read_image(img_path)
#         # 转回 PIL，兼容你现有 transform pipeline
#         img = to_pil_image(img)
#         return img
#     except:
#         # 失败时 fallback 到 PIL（不会报错）
#         try:
#             return Image.open(img_path).convert("RGB")
#         except:
#             print(f"IOError when reading {img_path}, retrying...")
#             return Image.open(img_path).convert("RGB")
def read_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, aux_info=False, transform=None):
        self.dataset = dataset
        self.aux_info = aux_info
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.aux_info:
            img_path, pid, camid, clothes_id, attr = self.dataset[index]
        else:
            img_path, pid, camid, clothes_id = self.dataset[index]

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.aux_info:
            # 如果以后真要用属性，这里再把 attr 转成 tensor
            attr = torch.tensor(np.asarray(attr, dtype=np.float32))
            return img, pid, camid, clothes_id, attr  # 5 个
        else:
            return img, pid, camid, clothes_id
# class ImageDataset(Dataset):
#     """Image Person ReID Dataset (supports aux_info)."""
#     def __init__(self, dataset, aux_info=False, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         self.aux_info = aux_info

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         if self.aux_info:
#             img_path, pid, camid, clothes_id, attr = self.dataset[index]
#             img = read_image(img_path)

#             if self.transform is not None:
#                 img = self.transform(img)

#             cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
#             attr = torch.tensor(np.asarray(attr, dtype=np.float32))
#             return img, pid, camid, clothes_id, cloth_id_batch, attr

#         else:
#             img_path, pid, camid, clothes_id = self.dataset[index]
#             img = read_image(img_path)

#             if self.transform is not None:
#                 img = self.transform(img)

#             cloth_id_batch = torch.tensor(clothes_id, dtype=torch.int64)
#             return img, pid, camid, clothes_id, cloth_id_batch


def pil_loader(path):
    """Simplified PIL loader."""
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def video_loader(img_paths):
    frames = []
    for path in img_paths:
        if osp.exists(path):
            frames.append(pil_loader(path))
        else:
            return frames
    return frames


class VideoDataset(Dataset):
    """Video Person ReID Dataset (remove accimage)."""
    def __init__(
        self,
        dataset,
        spatial_transform=None,
        temporal_transform=None,
        cloth_changing=True,
    ):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = video_loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid