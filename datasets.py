from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageData(Dataset):
    def __init__(self, files, labels=None, force_rgb=False, transform=None):
        self.files = files
        self.transform = transform
        self.labels = labels
        self.force_rgb = force_rgb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.files[index]))
        if len(image.shape) == 2 and self.force_rgb:
            image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(np.uint8(image))
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image, self.files[index]
        return image, self.labels[index], self.files[index]