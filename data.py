import os
from typing import Tuple
# from typing_extensions import override
import pandas as pd
from torchvision.datasets import DatasetFolder
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image

# def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
#     """Checks if a file is an allowed extension.

#     Args:
#         filename (string): path to a file
#         extensions (tuple of strings): extensions to consider (lowercase)

#     Returns:
#         bool: True if the filename ends with one of given extensions
#     """
#     return filename.lower().endswith(extensions)


# def is_image_file(filename: str) -> bool:
#     """Checks if a file is an allowed image extension.

#     Args:
#         filename (string): path to a file

#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     return has_file_allowed_extension(filename, IMG_EXTENSIONS)


# def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
#     """Finds the class folders in a dataset.

#     See :class:`DatasetFolder` for details.
#     """
#     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
#     if not classes:
#         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

#     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#     return classes, class_to_idx


# def make_dataset(
#     directory: str,
#     class_to_idx: Optional[Dict[str, int]] = None,
#     extensions: Optional[Tuple[str, ...]] = None,
#     is_valid_file: Optional[Callable[[str], bool]] = None,
# ) -> List[Tuple[str, int]]:
#     """Generates a list of samples of a form (path_to_sample, class).

#     See :class:`DatasetFolder` for details.

#     Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
#     by default.
#     """
#     directory = os.path.expanduser(directory)

#     if class_to_idx is None:
#         _, class_to_idx = find_classes(directory)
#     elif not class_to_idx:
#         raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

#     if extensions is not None:

#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

#     is_valid_file = cast(Callable[[str], bool], is_valid_file)

#     instances = []
#     available_classes = set()
#     for target_class in sorted(class_to_idx.keys()):
#         class_index = class_to_idx[target_class]
#         target_dir = os.path.join(directory, target_class)
#         if not os.path.isdir(target_dir):
#             continue
#         for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#             for fname in sorted(fnames):
#                 if is_valid_file(fname):
#                     path = os.path.join(root, fname)
#                     item = path, class_index
#                     instances.append(item)

#                     if target_class not in available_classes:
#                         available_classes.add(target_class)

#     empty_classes = set(class_to_idx.keys()) - available_classes
#     if empty_classes:
#         msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
#         if extensions is not None:
#             msg += f"Supported extensions are: {', '.join(extensions)}"
#         raise FileNotFoundError(msg)

#     return instances

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageNetCDataset(DatasetFolder):
    def __init__(
        self,
        root: str, # This should be directory before each corruption folder
        corruptions, # This should be a list of corruptions that getitem chooses from
        severity, # This should be 1,2,3,4,5
        loader: Callable[[str], Any]= default_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):

        super(ImageNetCDataset, self).__init__(
            root+'/'+corruptions[0]+'/'+str(severity),
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.root = root
        # self.imgs = self.samples
        # TODO: figure out a way to get 3 different self.samples
        self.cor_samples = []
        self.cor_samples.append(self.samples)
        for cor in corruptions[1:]:
            self.cor_samples.append(self.make_dataset(self.root+'/'+cor+'/'+str(severity), self.class_to_idx, self.extensions, is_valid_file))
        print("DEBUGGING: lengh of default self.samples =", len(self.samples))
        print("DEBUGGING: lengh of second corruption samples =", len(self.cor_samples[1]))
        print("DEBUGGING: the root of the imagefolder class is:", self.root)

    def __len__(self) -> int:
        assert len(self.cor_samples[0])==len(self.cor_samples[1])
        assert len(self.cor_samples[0])==len(self.cor_samples[1])
        return len(self.samples)

    # @override
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
        
        # can potentially can given a index, go into other folders to get an image.
        
        # I copied these from the ImageFolder source code
        samples = random.choice(self.cor_samples)
        path, target = samples[index]
        # TODO: QUESTION: what is path?? a path string? Can I modify this?
        print("DEBUGGING: printing the path:", path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # sample, target = ImageFolder.__getitem__(self, index)


        return sample, target

        # return

