
# function 'norm ' has been modified. Please keep a check of it.
import numpy as np
from lightly.data import LightlyDataset
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split


from ..data_utils import encode_labels, load_npzs
from ..utils import get_logger

import torch
from torch.utils.data import Dataset
import glob

class GaussianNoise(T._transform.Transform):
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.

    """
    def __init__(self, p=0.5, sigma=(6, 10)):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if torch.rand(1)[0] < self.p:
            mu = sample.to(float).mean()
            snr = self.sigma
            if isinstance(self.sigma, tuple):
                snr = torch.randint(low=self.sigma[0], high=self.sigma[1], size=(1,))
            sigma = mu.abs() / snr
            noise = torch.normal(torch.zeros(sample.shape), sigma)
            if sample.dtype == torch.uint8:
                noise = noise.to(torch.uint8)
            return sample + noise
        return sample


class Norm(T._transform.Transform):
    """Independently normalize images"""

    def __init__(self, scale=True):
        self.scale = scale

    @staticmethod
    def T(t):
        if t.ndim == 2:
            return t.T
        elif t.ndim == 1 or t.ndim == 0:
            return t
        else:
            return t.permute(*torch.arange(t.ndim - 1, -1, -1))

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        ret = self.T(self.T(sample) - self.T(sample.mean(dim=(-2, -1))))
        if self.scale:
            ret = self.T(self.T(ret) / self.T(torch.std(ret, dim=(-2, -1))))
        return ret


class LMDataset(Dataset):
    """

    Properties:
        data (Tensor)                          : images
        sample_labels (list of Tensors)        : labels for each sample
        paths (tuple)                          : original path of each image
        transform (Transform)                  : transform to apply before returning images
        label (list)                           : a list of one or more labels to use for supervised learning

    """

    __regression_labels = {'time'}

    @classmethod
    def is_regression(cls, label):
        return label in cls.__regression_labels

    def __init__(self, npzs, label_classes=None, use_masks=False, return_labels=False, logger=None,
                 transform=None, label=None, n_samples=None, split=None, val_size=None, seed=None):
        """
        Args:
            npzs (array-like)           : A list or tuple of paths to NPZ files containing cropped images
            label_classes (dict)        : a dictionary of classes for each label
            use_masks (bool)            : whether or not to use masks instead of images
            return_labels (bool)        : whether or not to return labels for each data point
            logger (Logger)             : the logger to use when loading data
            transform (Transform)       : the transform to apply to each image
            label (str, list, tuple)    : the label(s) to return when getting a sample
            n_samples (int)             : the number of samples to load from each file
            split (str)                 : a string indication which split to use ("train" or "val")
            val_size (float)            : fraction of data to use for validation
            seed (int)                  : the RNG seed to use for training/validation split

        """
        if not isinstance(npzs, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(f"Got unexpected type ({type(npzs)}) for argument 'npzs'. Must be an array-like")
        elif len(npzs) == 0:
            raise ValueError("Got empty array-like for argument 'npzs'")
        logger = logger or get_logger('warning')

        masks, images, paths, metadata = load_npzs(npzs, logger, n_samples, label)
        if use_masks:
            self.data = masks
        else:
            self.data = images
        self.data = torch.from_numpy(self.data)[:, None, :, :]
        self.paths = tuple(paths)
        self.transform = transform

        if not isinstance(label, (tuple, list)):
            label = [label]

        self.sample_labels = None
        # self.label_classes is set up so that we can specify label_classes or compute them on the fly.
        # We want to specify them after they have been computed to ensure the same label classes are
        # used across splits (e.g. training, validation, test sets)
        self.label_classes = label_classes or dict()
        self.label = label

        if return_labels:
            self.sample_labels = []
            for k in self.label:
                if self.is_regression(k):
                    labels = torch.from_numpy(encode_labels(metadata[k], 'regression'))
                    self.label_classes[k] = None
                else:
                    labels_np, classes = encode_labels(metadata[k], 'classification', classes=self.label_classes.get(k), return_classes=True)
                    labels = torch.from_numpy(labels_np)
                    self.label_classes[k] = classes
                self.sample_labels.append(labels)

        self.metadata = metadata

        if val_size:
            self._split_data(split, val_size, seed)

    def _split_data(self, split, val_size, seed):
        num_samples = len(self.data)
        indices = np.arange(num_samples)

        if self.sample_labels is not None:
            stratify_label = np.stack([label.numpy() for label in self.sample_labels], axis=1)
            composite_label = [tuple(row) for row in stratify_label]

            train_indices, val_indices = train_test_split(
                indices, test_size=val_size, random_state=seed, stratify=composite_label
            )
        else:
            train_indices, val_indices = train_test_split(
                indices, test_size=val_size, random_state=seed
            )

        if split == 'train':
            self.data = self.data[train_indices]
            if self.sample_labels is not None:
                self.sample_labels = [label[train_indices] for label in self.sample_labels]
        elif split.startswith('val'):
            self.data = self.data[val_indices]
            if self.sample_labels is not None:
                self.sample_labels = [label[val_indices] for label in self.sample_labels]

    def __getitem__(self, i):
        ret = self.data[i]
        if self.transform is not None:
            ret = self.transform(ret)
        if self.sample_labels is None:
            return ret
        else:
            ret_tmp = [self.sample_labels[j][i] for j in range(len(self.sample_labels))]
            return ret, tuple(ret_tmp)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def index_to_filename(dataset, i):
        return dataset.paths[i]

def extract_labels_from_filename(filename):
    """Extract labels from filename in the format Sx_HTY_randomtext.npz"""
    parts = filename.split('/')[-1].split('_')
    x_label = parts[0][1:]  # Extract X from SX
    y_label = parts[1][2:]  # Extract Y from HTY
    return x_label, y_label

TRANSFORMS = {
        'blur': T.GaussianBlur(3, sigma=(0.01, 1.0)),
        'rotate': T.RandomRotation(180),
        'crop': T.CenterCrop((64, 64)),
        'hflip': T.RandomHorizontalFlip(0.5),
        'vflip': T.RandomVerticalFlip(0.5),
        'noise': GaussianNoise(sigma=(10, 12)),
        'rgb': T.Lambda(lambda x: x.repeat(3, 1, 1) if x.ndim == 3 else x.repeat(1, 3, 1, 1)),
        'float': T.Lambda(lambda x: x.to(torch.float32)),
        'norm': Norm(),
}


def get_transforms(*transforms):
    """Return a transforms appropriate for Ambr light microscopy data

    The following transforms and their respective keys are:
        norm:       Normalize the image by subtracting the mean pixel value from
                    image
        blur:       A Gaussian Blur
        rotate:     Random rotation of up to 180 degrees in either direction
        crop:       Center crop images to 64x64 pixels
        hflip:      Random horizontal rotation with probability of 0.5
        vflip:      Random vertical rotation with probability of 0.5
        noise:      Randomly apply noise with probability of 0.5. Noise magnitude
                    will be between 10-12 signal-to-noise ratio
        rgb:        Convert to RGB format i.e. 3 channels
        float:      Convert to torch.Float dtype

    Args:
        transforms: the list of transforms to get. Valid options are 'blur', 'rotate', 'crop',
                    'hflip', 'vflip', 'noise', 'rgb', 'float', 'norm'
    Returns:
        a single transform or a Compose object pipeline with transforms in the order they
        are given. If no transforms are specified, the identity transform will be returned
    """
    if len(transforms) == 0:
        return lambda x: x
    ret = list()
    for tfm in transforms:
        if tfm not in TRANSFORMS:
            raise ValueError(f"Unrecognozed transform: '{tfm}'")
        ret.append(TRANSFORMS[tfm])
    return ret[0] if len(ret) == 1 else T.Compose(ret)



def get_lightly_dataset(npzs, transform=None, **lmdset_kwargs):
    """Helper function for getting a LightlyDataset"""
    dataset = LMDataset(npzs, **lmdset_kwargs)
    return LightlyDataset.from_torch_dataset(dataset,
                                             transform=transform,
                                             index_to_filename=dataset.index_to_filename)
