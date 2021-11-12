from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
import torch
from torchvision.transforms import functional as F


class ScaledCenterCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        new_size = (int(img.size[0] * self.factor),int(img.size[1] * self.factor))
        return F.center_crop(img, new_size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterFullCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        new_size = min(img.size[0], img.size[1])
        return F.center_crop(img, (new_size,new_size))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class TopTracker():
    def __init__(self, top_k=5, extrema="min"):
        self.items = []
        self.scores = []
        self.k = top_k
        self.extrema = extrema

        self.test = lambda a, b: a < b

    def get_top_items(self):
        if self.extrema == "min":
            sorted_idx = np.argsort(self.scores)
            return [self.items[i] for i in sorted_idx]
        elif self.extrema == "max":
            sorted_idx = np.argsort(self.scores)[::-1]
            return [self.items[i] for i in sorted_idx]

    def get_top_scores(self):
        if self.extrema == "min":
            return np.sort(self.scores)
        elif self.extrema == "max":
            return np.sort(self.scores)[::-1]

    def add(self, score, item):
        if len(self.items) < self.k:
            self.items.append(item)
            self.scores.append(score)
        else:
            if self.extrema == "min":
                max_score_ind = np.argmax(self.scores)
                if self.scores[max_score_ind] > score:
                    self.scores[max_score_ind] = score
                    self.items[max_score_ind] = item
            elif self.extrema == "max":
                min_score_ind = np.argmin(self.scores)
                if self.scores[min_score_ind] > score:
                    self.scores[min_score_ind] = score
                    self.items[min_score_ind] = item

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def slit_data(df, test_split, val_split, seed=7):
    indices = np.array(range(df.shape[0]))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_point_1 = int(indices.shape[0] * test_split)
    split_point_2 = int(indices.shape[0] * (val_split + test_split))
    test_indices = indices[0:split_point_1]
    val_indices = indices[split_point_1:split_point_2]
    train_indices = indices[split_point_2::]
    train_df = df.take(train_indices)
    test_df = df.take(test_indices)
    val_df = df.take(val_indices)
    return train_df, test_df, val_df


def get_train_transform_sd(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_train_transform_isic(input_size):
    """

    :return:
    """

    return transforms.Compose([
        CenterFullCrop(),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_test_transform_isic(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_test_transform_sd(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_isic(isic2019csv, test_split, val_split):
    df = pd.read_csv(os.path.join(isic2019csv))
    train_df, test_df, val_df = slit_data(df, test_split, val_split)
    image_dir = os.path.join(os.path.dirname(isic2019csv), "ISIC_2019_Training_Input")
    train_files = [os.path.join(image_dir, f + ".jpg") for f in train_df.image]
    train_labels = np.argmax(train_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    val_files = [os.path.join(image_dir, f + ".jpg") for f in val_df.image]
    val_labels = np.argmax(val_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    test_files = [os.path.join(image_dir, f + ".jpg") for f in test_df.image]
    test_labels = np.argmax(test_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def get_directory(negative_dir, test_split, val_split):
    result = [y for x in os.walk(negative_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    df = pd.DataFrame({"images": result})
    train_df, test_df, val_df = slit_data(df, test_split, val_split)
    return train_df["images"].tolist(), test_df["images"].tolist(), val_df["images"].tolist()