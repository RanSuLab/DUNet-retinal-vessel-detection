from torch.utils import data
from torchvision.transforms import ToTensor
import numpy as np


class Retina_loader(data.Dataset):
    def __init__(self, patches_imgs_train, patches_masks_train, split_radio=0.9, split="train", fcn=True):
        self.split = split
        self.fcn = fcn
        if split == "train":
            self.patches_imgs_train = patches_imgs_train[0:int(len(patches_imgs_train) * split_radio)]
            self.patches_masks_train = patches_masks_train[0:int(len(patches_imgs_train) * split_radio)]
        if split == "test":
            self.patches_imgs_train = patches_imgs_train[int(len(patches_imgs_train) * split_radio):]
            self.patches_masks_train = patches_masks_train[int(len(patches_imgs_train) * split_radio):]
        print('number of ' + split + ': ' + str(len(self.patches_imgs_train)))

    def __len__(self):
        return len(self.patches_imgs_train)

    def __getitem__(self, index):
        image = self.patches_imgs_train[index]
        label = self.patches_masks_train[index]
        if self.fcn:
            return ToTensor()(image), ToTensor()(label)
        else:
            return ToTensor()(image), np.argmax(label)


class Retina_loader_infer(data.Dataset):
    def __init__(self, patches_imgs_test):
        self.patches_imgs_test = patches_imgs_test

    def __len__(self):
        return len(self.patches_imgs_test)

    def __getitem__(self, index):
        image = self.patches_imgs_test[index]
        return ToTensor()(image)
