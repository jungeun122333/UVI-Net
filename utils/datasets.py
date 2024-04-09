import copy
import glob
import os
import pickle
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


class ACDCHeartDataset(Dataset):
    def __init__(self, data_path, phase="Train", split=90):
        self.path = data_path
        self.fine_size = (128, 128, 32)
        if phase == "train":
            self.paths = sorted(os.listdir(self.path))[1 : split + 1]
        elif phase == "test":
            self.paths = sorted(os.listdir(self.path))[split + 1 :]

    def __getitem__(self, index):
        patient_folder = os.path.join(self.path, self.paths[index])
        with open(f"{patient_folder}/Info.cfg", "rt") as f:
            ED = int(f.readline().strip().split()[-1])
            ES = int(f.readline().strip().split()[-1])

        ED_image = nib.load(
            f"{patient_folder}/{self.paths[index]}_frame{ED:02d}.nii.gz"
        ).get_fdata()
        ES_image = nib.load(
            f"{patient_folder}/{self.paths[index]}_frame{ES:02d}.nii.gz"
        ).get_fdata()
        video = nib.load(f"{patient_folder}/{self.paths[index]}_4d.nii.gz").get_fdata()

        nh, nw, nd = ED_image.shape
        fh, fw, fd = self.fine_size
        sh = (nh - fh) // 2
        sw = (nw - fw) // 2
        sd = (nd - fd) // 2

        ED_image = ED_image[sh : sh + fh, sw : sw + fw]
        ES_image = ES_image[sh : sh + fh, sw : sw + fw]
        video = video[sh : sh + fh, sw : sw + fw]

        if nd >= fd:
            ED_image = ED_image[..., sd : sd + fd][None, ...]
            ES_image = ES_image[..., sd : sd + fd][None, ...]
            video = video[..., sd : sd + fd, :][None, ...]
        else:
            zeros_image1 = np.zeros((fh, fw, (fd - nd) // 2))
            zeros_image2 = np.zeros((fh, fw, fd - nd - (fd - nd) // 2))
            zeros_video1 = np.zeros((fh, fw, (fd - nd) // 2, video.shape[-1]))
            zeros_video2 = np.zeros((fh, fw, fd - nd - (fd - nd) // 2, video.shape[-1]))

            ED_image = np.concatenate([zeros_image1, ED_image, zeros_image2], axis=-1)[
                None, ...
            ]
            ES_image = np.concatenate([zeros_image1, ES_image, zeros_image2], axis=-1)[
                None, ...
            ]
            video = np.concatenate([zeros_video1, video, zeros_video2], axis=-2)[
                None, ...
            ]

        # 0~1 scaling
        ED_image = (ED_image - ED_image.min()) / (ED_image.max() - ED_image.min())
        ES_image = (ES_image - ES_image.min()) / (ES_image.max() - ES_image.min())
        for i in range(video.shape[-1]):
            video[..., i] = (video[..., i] - video[..., i].min()) / (
                video[..., i].max() - video[..., i].min()
            )

        ED_image = torch.from_numpy(ED_image).float()
        ES_image = torch.from_numpy(ES_image).float()
        video = torch.from_numpy(video).float()

        return ED_image, ES_image, ED - 1, ES - 1, video

    def __len__(self):
        return len(self.paths)


class LungDataset(Dataset):
    def __init__(self, data_path, phase, split=68):
        self.path = data_path
        self.fine_size = (128, 128, 128)
        if phase == "train":
            self.paths = sorted(os.listdir(self.path))[:split]
        elif phase == "test":
            self.paths = sorted(os.listdir(self.path))[split:]

    def __getitem__(self, index):
        patient_folder = os.path.join(self.path, self.paths[index])
        ED = 0
        ES = 5

        ED_image = nib.load(
            f"{patient_folder}/ct_{self.paths[index]}_frame{ED}.nii.gz"
        ).get_fdata()
        ES_image = nib.load(
            f"{patient_folder}/ct_{self.paths[index]}_frame{ES}.nii.gz"
        ).get_fdata()

        ED_image = ED_image[None, ...]
        ES_image = ES_image[None, ...]
        video = copy.deepcopy(ED_image)[..., None]
        for idx in range(1, ES + 1):
            frame_img = nib.load(
                f"{patient_folder}/ct_{self.paths[index]}_frame{idx}.nii.gz"
            ).get_fdata()
            frame_img = frame_img[None, ..., None]
            video = np.concatenate([video, frame_img], axis=-1)

        ED_image = torch.from_numpy(ED_image).float()
        ES_image = torch.from_numpy(ES_image).float()
        video = torch.from_numpy(video).float()

        return ED_image, ES_image, ED, ES, video

    def __len__(self):
        return len(self.paths)
