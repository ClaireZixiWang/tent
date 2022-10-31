import os
from typing import Tuple
import pandas as pd
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split


class ImageNetCDataset(ImageFolder):
    # def __init__(
    #     self,
    #     root: str,
    #     transform: Optional[Callable] = None,
    #     target_transform: Optional[Callable] = None,
    #     loader: Callable[[str], Any] = default_loader,
    #     is_valid_file: Optional[Callable[[str], bool]] = None,
    # ):
    #     super().__init__(
    #         root,
    #         loader,
    #         IMG_EXTENSIONS if is_valid_file is None else None,
    #         transform=transform,
    #         target_transform=target_transform,
    #         is_valid_file=is_valid_file,
    #     )
    #     self.imgs = self.samples

    # def __len__(self):

    #     # len
    #     return 

    def __getitem__(self, index: int):#, mixed: bool):  # type: ignore

        # the original one
        # if not mixed:
            # sample, target = super.__getitem__(index)


        # also need to write the one with mixed corruption
        # I don't think you can modify getitem that much, especially since you cannot write your own
        # DataLoader (because you don't know what you are doing), you can't plug in an extra argument
        # A cleaner solution would be to contruct a MixedImageNetCDataset Class, that can retrieve
        # corruptions from different folders
        # else:
        sample, target = ImageFolder.__getitem__(self, index)


        return sample, target

        # return

