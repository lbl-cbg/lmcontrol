import sys

# function 'norm ' has been modified. Please keep a check of it.
import numpy as np
from lightly.data import LightlyDataset
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split
from hdmf.common import get_hdf5io

from ..data_utils import encode_labels, load_npzs
from ..utils import get_logger

import torch
from torch.utils.data import Dataset

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

    split_vals = {'train': 0, 'validation': 1, 'test': 2}

    @classmethod
    def is_regression(cls, label):
        return label in cls.__regression_labels

    def __init__(self, path, label_classes=None, use_masks=False, return_labels=False, logger=None, transform=None,
                 label=None, n_samples=None, split=None, rand_split=False, exp_split=False, split_seed=None):
        """
        Args:
            path (str)                  : path to the HDMF DynamicTable file
            label_classes (dict)        : a dictionary of classes for each label
            use_masks (bool)            : whether or not to use masks instead of images
            return_labels (bool)        : whether or not to return labels for each data point
            logger (Logger)             : the logger to use when loading data
            transform (Transform)       : the transform to apply to each image
            label (str, list, tuple)    : the label(s) to return when getting a sample
            n_samples (int)             : the number of samples to load from each file
            split (str)                 : a string indication which split to use ("train" or "val")
            exp_split (bool)            : calculate train-validation-test split using metdata on campaign, condition,
                                          and ht (a.k.a. replicate). validation and test split are first calculated
                                          by getting single unique ht for each condition for each campaign. This is
                                          then split between campaigns. All remaining samples are left for training.
                                          exp_split is short for "experimental split"

        """
        import uuid
        self.uuid = uuid.uuid1()
        if rand_split and exp_split:
            raise ValueError("rand_split and exp_split cannot both be True")

        self.logger = logger or get_logger('warning')

        self.path = path

        self.use_masks = use_masks
        self.return_labels = return_labels

        if not isinstance(label, (tuple, list)):
            label = [label]

        self.label = label
        self.sample_labels = None

        self.transform = transform

        self.split = split
        self.split_mask = None
        self.subset = None

        self.io = None
        self.table = None

        self.open()
        self.__len = len(self.table)

        self.exp_split = exp_split
        self.rand_split = rand_split

        if exp_split:
            train_mask, val_mask, test_mask = make_masks(self.table, split_validation=True, seed=split_seed)
            split_mask = np.zeros(self.__len, dtype=int)
            split_mask[train_mask] = self.split_vals['train']
            split_mask[val_mask] = self.split_vals['validation']
            split_mask[test_mask] = self.split_vals['test']
            self.set_split(split_mask)
        elif rand_split:
            self.set_random_split(0.1, 0.1, seed=split_seed)

        self.close()

        # self.label_classes is set up so that we can specify label_classes or compute them on the fly.
        # We want to specify them after they have been computed to ensure the same label classes are
        # used across splits (e.g. training, validation, test sets)
        self.label_classes = label_classes or dict()

    def open(self, worker_id=None):
        """Open file for reading if it is not currently open"""
        if self.table is None:
            self.io = get_hdf5io(self.path, 'r')
            self.table = self.io.read()

            if self.use_masks:
                self.data = self.table['masks'].data
            else:
                self.data = self.table['images'].data

            self.label_classes = dict()

            if self.return_labels:
                self.sample_labels = []
                for k in self.label:
                    if isinstance(self.table[k], EnumData):
                        self.label_classes[k] = self.table[k].elements[:]
                    else:
                        self.label_classes[k] = None
                    self.sample_labels.append(self.table[k].data)

    def close(self):
        self.io.close()
        self.table = None
        self.io = None
        self.data = None

    def set_random_split(self, val_frac, test_frac, seed=None):
        if (val_frac + test_frac) >= 1.0:
            raise ValueError("val and test should sum to less 1.0, otherwise there will be nothing left to train with!")

        # open the file if we need to, but make sure we close it if it was closed to begin with
        close = self.table is None
        if close:
            self.open()

        n = len(self.table)
        val_len = int(n * val_frac)
        test_len = int(n * test_frac)
        train_len = n - val_len - test_len

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        split = torch.zeros(n, dtype=torch.uint8) + self.split_vals['train']
        split[train_len:train_len+val_len] = self.split_vals['validation']
        split[train_len+val_len:] = self.split_vals['test']

        split = split[torch.randperm(n, generator=rng)]

        self.set_split(split)

        if close:
            self.close()

    def set_split(self, split_mask):
        self.split_mask = torch.as_tensor(split_mask)
        if self.split is not None:
            self.subset = torch.where(self.split_mask == self.split_vals[self.split])[0]
            self.__len = len(self.subset)


    def __getitem__(self, i):
        i = i if self.subset is None else self.subset[i]
        ret = torch.as_tensor(self.data[i]).unsqueeze(0)
        if self.transform is not None:
            ret = self.transform(ret)
        if self.sample_labels is None:
            return ret
        else:
            ret_tmp = [torch.as_tensor(self.sample_labels[j][i])
                       for j in range(len(self.sample_labels))]
            return ret, tuple(ret_tmp)

    def __len__(self):
        return self.__len

    @staticmethod
    def index_to_filename(dataset, i):
        return dataset.paths[i]


def extract_labels_from_filename(filename):
    """Extract labels from filename in the format Sx_HTY_randomtext.npz"""
    parts = filename.split('/')[-1].split('_')
    x_label = parts[0][1:]  # Extract X from SX
    y_label = parts[1][2:]  # Extract Y from HTY
    return x_label, y_label


def to_rgb(x):
    return x.repeat(3, 1, 1) if x.ndim == 3 else x.repeat(1, 3, 1, 1)

def to_float(x):
    return x.to(torch.float32)

TRANSFORMS = {
        'blur': T.GaussianBlur(3, sigma=(0.01, 1.0)),
        'rotate': T.RandomRotation(180),
        'crop': T.CenterCrop((64, 64)),
        'hflip': T.RandomHorizontalFlip(0.5),
        'vflip': T.RandomVerticalFlip(0.5),
        'noise': GaussianNoise(sigma=(10, 12)),
        'rgb': T.Lambda(to_rgb),
        'float': T.Lambda(to_float),
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


def make_masks(table, split_validation=False, validation_split_ratio=0.5, seed=None):
    """
    Creates masks for training and validation splits using only numpy.

    Args:
        table: Dictionary-like object containing the data arrays
        split_validation: If True, splits validation set into two separate sets
        validation_split_ratio: Ratio for splitting validation campaigns (default: 0.5)

    Returns:
        train_mask: Boolean array for training data
        val_mask: Boolean array for validation data (or first validation set if split)
        val2_mask: Boolean array for second validation set (if split_validation=True)
        rep_tuples: List of representative tuples (campaign, condition, ht) in validation
        rep_tuples2: List of representative tuples in second validation set (if split_validation=True)
    """
    mask = table['source'].data[:] == 1

    campaign = table['campaign'].data[:][mask]
    condition = table['condition'].data[:][mask]
    ht = table['ht'].data[:][mask]

    # Get unique campaigns and potentially split them
    unique_campaigns = np.sort(np.unique(campaign))

    if split_validation:
        # Randomly shuffle campaigns for splitting
        np.random.default_rng(seed).shuffle(unique_campaigns)
        split_idx = int(len(unique_campaigns) * validation_split_ratio)
        val1_campaigns = unique_campaigns[:split_idx]
        val2_campaigns = unique_campaigns[split_idx:]
    else:
        val1_campaigns = unique_campaigns
        val2_campaigns = []

    def process_validation_set(campaigns):
        """
        Helper function to process a set of campaigns for validation
        """
        indices = []

        for camp in campaigns:
            camp_mask = campaign == camp
            camp_conditions = np.unique(condition[camp_mask])

            for cond in camp_conditions:
                cond_mask = (campaign == camp) & (condition == cond)
                unique_hts = np.sort(np.unique(ht[cond_mask]))

                if len(unique_hts) > 0:
                    selected_ht = unique_hts[0]

                    val_indices_for_cond = np.where((campaign == camp) &
                                                  (condition == cond) &
                                                  (ht == selected_ht))[0]
                    indices.extend(val_indices_for_cond)

        return indices

    # Process validation sets
    val1_indices = process_validation_set(val1_campaigns)

    if split_validation:
        val2_indices = process_validation_set(val2_campaigns)

    # Create the masks
    tmp_val1_mask = np.zeros(len(campaign), dtype=bool)
    tmp_val1_mask[val1_indices] = True

    indices = np.where(mask)[0]
    val1_mask = np.zeros_like(mask)
    val1_mask[indices[tmp_val1_mask]] = True

    if split_validation:
        tmp_val2_mask = np.zeros(len(campaign), dtype=bool)
        tmp_val2_mask[val2_indices] = True

        val2_mask = np.zeros_like(mask)
        val2_mask[indices[tmp_val2_mask]] = True

        # Training mask excludes both validation sets
        train_mask = np.zeros_like(mask)
        train_mask[indices[~(tmp_val1_mask | tmp_val2_mask)]] = True

        return train_mask, val1_mask, val2_mask
    else:
        # Training mask excludes validation set
        train_mask = np.zeros_like(mask)
        train_mask[indices[~tmp_val1_mask]] = True

        return train_mask, val1_mask
