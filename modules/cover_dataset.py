import os
import glob
from typing import List
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image


def get_valid_videos(image_root):
    valid_names = {}
    for name in os.listdir(image_root):
        sub_root = os.path.join(image_root, name)
        if not os.path.isdir(sub_root):
            continue
        image_files = glob.glob(os.path.join(sub_root, "*.jpg"))
        if len(image_files) != 10:
            continue
        image_files = sorted(
            image_files,
            key=lambda name: int(os.path.splitext(os.path.basename(name))[0]),
        )
        valid_names[name] = image_files
    return valid_names


def read_and_merge_datasets(image_roots, label_files):
    all_label_data = []
    valid_videos = {}
    for image_root, label_file in zip(image_roots, label_files):
        all_label_data.append(pd.read_csv(label_file))
        valid_videos.update(get_valid_videos(image_root))
    df = pd.concat(all_label_data, axis=0, ignore_index=True)

    # filter based on impr_cnt
    impr_cnt = np.asarray(df["impr_cnt"], dtype=int)
    df = df[impr_cnt >= 10]

    label_names = ["click_ratio", "like_ratio", "play_finish_ratio"]
    labels = np.zeros([len(df), len(label_names)], dtype=int)
    for i, name in enumerate(label_names):
        threshold = np.median(df[name])
        labels[:, i] = df[name] > threshold

    # filter based on image
    valid_labels = []
    for vid, label in zip(df["vid"], labels):
        if vid in valid_videos:
            valid_labels.append((vid, label))

    # sort by vid
    valid_labels = sorted(valid_labels, key=lambda item: item[0])

    return valid_labels, valid_videos


class CoverDataset(Dataset):
    def __init__(
        self,
        image_roots,
        label_files,
        subset="train",
        tranform=None,
        image_loader=default_loader,
        add_blank=0,
        statistics=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.transform = tranform
        self.image_loader = image_loader
        self.add_blank = add_blank
        self.statistics = (torch.Tensor(statistics[0]), torch.Tensor(statistics[1]))

        norm = transforms.Normalize(mean=statistics[0], std=statistics[1])
        if self.transform is not None:
            self.transform = transforms.Compose([self.transform, norm])
        else:
            self.transform = norm

        labels, image_names = read_and_merge_datasets(image_roots, label_files)
        self.image_names = image_names

        split = int(len(labels) * 0.8)
        np.random.seed(1234)
        np.random.shuffle(labels)
        if subset == "train":
            self.labels = labels[0:split]  # [(vid, labels)]
        else:
            self.labels = labels[split:]

        raw_labels = np.asarray([item[1] for item in self.labels])
        pos_rate = np.mean(raw_labels > 0, axis=0)
        print(
            "Collect {} videos for {}, positive rate for each target: {}".format(
                len(self.labels), subset, pos_rate
            )
        )

    def __len__(self):
        return len(self.labels) + self.add_blank

    def __getitem__(self, i):
        i -= self.add_blank

        if i >= 0:
            images, label = self.read_images(i)
        else:
            color = (255, 255, 255)
            if i == -1:
                color = (255, 255, 255)
            elif i == -2:
                color = (0, 0, 0)
            else:
                color = (np.random.randint(0, 255)) * 3
            images, label = self.get_black_images(color)

        images = torch.stack(images, 0)
        return images, int(label[0])

    def read_images(self, i):
        vid, label = self.labels[i]
        image_names = self.image_names[vid]

        # read images
        images = []  # type: List[torch.Tensor]
        for fullname in image_names:
            image = self.image_loader(fullname)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images, label

    def get_black_images(self, color=(255, 255, 255)):
        image = np.zeros([100, 100, 3], dtype=np.uint8)
        image[:, :] = color
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        label = np.asarray([0, 0, 0], dtype=int)
        return [image] * 10, label
