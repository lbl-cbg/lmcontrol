import sys
import warnings

# function 'norm ' has been modified. Please keep a check of it.
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split
from hdmf.common import get_hdf5io, EnumData

from ..data_utils import encode_labels, load_npzs
from ..utils import get_logger, import_ml

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

KernelDensity = import_ml("neighbors.KernelDensity")


class RGB(nn.Module):
    """Convert image to three channels"""

    @staticmethod
    def to_rgb(image):
        return image.repeat(3, 1, 1) if image.ndim == 3 else image.repeat(1, 3, 1, 1)

    def forward(self, image, mask=None):
        if mask is None:
            return self.to_rgb(image)
        return self.to_rgb(image), mask


class Float(nn.Module):
    """Convert to 32-bit floating point"""

    def forward(self, image, mask=None):
        if mask is None:
            return image.to(torch.float32)
        return image.to(torch.float32), mask


class GaussianBlur(nn.Module):
    """Performs Gaussian blurring on the image by given kernel"""
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_min = sigma[0]
        self.sigma_max = sigma[1]

    def forward(self, image, mask=None):
        sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
        ret = TF.gaussian_blur(image, self.kernel_size, [sigma, sigma])
        if mask is None:
            return ret
        return ret, mask


class GaussianNoise(nn.Module):
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.
    """
    def __init__(self, p=0.5, sigma=(6, 10)):
        super().__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, image, mask=None):
        ret = image
        if torch.rand(1)[0] < self.p:
            mu = image.to(float).mean()
            snr = self.sigma
            if isinstance(self.sigma, tuple):
                snr = torch.randint(low=self.sigma[0], high=self.sigma[1], size=(1,))
            sigma = mu.abs() / snr
            noise = torch.normal(torch.zeros(image.shape), sigma)
            if image.dtype == torch.uint8:
                noise = noise.to(torch.uint8)
            ret = image + noise
        if mask is None:
            return ret
        return ret, mask


class Norm(nn.Module):
    """Independently normalize images"""

    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale

    @staticmethod
    def T(t):
        if t.ndim == 2:
            return t.T
        elif t.ndim == 1 or t.ndim == 0:
            return t
        else:
            return t.permute(*torch.arange(t.ndim - 1, -1, -1))

    def forward(self, image, mask=None):
        ret = self.T(self.T(image) - self.T(image.mean(dim=(-2, -1))))
        if self.scale:
            ret = self.T(self.T(ret) / self.T(torch.std(ret, dim=(-2, -1))))
        if mask is None:
            return ret
        return ret, mask


class MaskNorm(nn.Module):
    """
    Normalizes images using statistics calculated only from background pixels
    as identified by the segmentation mask.
    """

    def __init__(self, eps=1e-8):
        """
        Args:
            eps (float): Small constant added to std to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.eps = eps

    def forward(self, image, mask=None):
        """
        Args:
            image (Tensor): Image to be normalized, shape [C, H, W]
            mask (Tensor): Segmentation mask, shape [1, H, W] or [H, W]

        Returns:
            Tensor, Tensor: Normalized image and unchanged mask
        """
        # Ensure mask has the right shape (add channel dim if needed)
        if image.dim() == 2:
            image = image.unsqueeze(0)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Initialize result tensor
        normalized_image = torch.zeros_like(image, dtype=torch.float)

        # Normalize each channel separately
        for c in range(image.shape[0]):
            channel = image[c].to(torch.float)
            ch_mask = mask[c].to(torch.float)
            background_mask = (ch_mask == 0).nonzero()
            # Extract background pixels for this channel
            background_pixels = channel[background_mask]

            # Check if we have enough background pixels
            if background_pixels.numel() > 0:
                # Calculate mean and std of background pixels
                mean = background_pixels.mean()
                std = background_pixels.std() + self.eps
            else:
                # Fallback if no background pixels found
                mean = channel.mean(dtype=torch.float)
                std = channel.std(dtype=torch.float) + self.eps

            # Normalize the entire channel using background statistics
            normalized_image[c] = (channel - mean) / std
        return normalized_image, mask


class RandomHorizontalFlip(nn.Module):
    """Randomly flip images horizontally with a given probability."""

    def __init__(self, p=0.5, generator=None):
        """
        Args:
            p (float): Probability of applying the horizontal flip. Default: 0.5
            generator (torch.Generator, optional): Generator for random number sampling
        """
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, image, mask=None):
        """
        Args:
            image (Tensor): Image to be flipped
            mask (Tensor): Mask to be flipped

        Returns:
            Tensor, Tensor: Transformed image and mask
        """
        flip = torch.rand(1, generator=self.generator).item() < self.p

        if flip:
            image = TF.hflip(image)
        if mask is None:
            return image
        if flip:
            mask = TF.hflip(mask)
        return image, mask


class RandomVerticalFlip(nn.Module):
    """Randomly flip images vertically with a given probability."""

    def __init__(self, p=0.5, generator=None):
        """
        Args:
            p (float): Probability of applying the vertical flip. Default: 0.5
            generator (torch.Generator, optional): Generator for random number sampling
        """
        super().__init__()
        self.p = p
        self.generator = generator

    def forward(self, image, mask=None):
        """
        Args:
            image (Tensor): Image to be flipped
            mask (Tensor): Mask to be flipped

        Returns:
            Tensor, Tensor: Transformed image and mask
        """
        flip = torch.rand(1, generator=self.generator).item() < self.p

        if flip:
            image = TF.vflip(image)
        if mask is None:
            return image
        if flip:
            mask = TF.vflip(mask)
        return image, mask



class RandomRotation(nn.Module):
    """Randomly rotate images within a given range of degrees."""

    def __init__(self, max_degrees, generator=None):
        """
        Args:
            max_degrees (float): Maximum rotation angle in degrees.
                                 Will sample uniformly between 0 and max_degrees.
            generator (torch.Generator, optional): Generator for random number sampling
        """
        super().__init__()
        self.max_degrees = max_degrees
        self.generator = generator

    def forward(self, image, mask=None):
        """
        Args:
            image (Tensor): Image to be rotated
            mask (Tensor): Mask to be rotated

        Returns:
            Tensor, Tensor: Transformed image and mask
        """
        angle = torch.rand(1, generator=self.generator).item() * self.max_degrees

        # Apply the same rotation to both image and mask
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        if mask is None:
            return image
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.BILINEAR)
        return image, mask


class RandomCrop(nn.Module):

    def __init__(self, size, generator=None):
        """
        Args:
            generator (torch.Generator, optional): Generator for random number sampling
        """
        super().__init__()
        self.size = size
        self.generator = generator

    @classmethod
    def compute_upper_H_W(cls, mask):
        # Get indices of non-zero elements
        indices = torch.nonzero(mask, as_tuple=True)

        # Calculate center as mean of indices
        if indices[0].numel() > 0:
            Xx, Xn = indices[0].max(), indices[0].min()
            Yx, Yn = indices[1].max(), indices[1].min()
            h = Xx - Xn
            w = Yx - Yn
            return Xn, Yn, h, w
        else:
            # Return center of image if mask is empty
            h, w = mask.shape
            return 0, 0, h, w

    def forward(self, image, mask):
        X, Y, H, W = self.compute_upper_H_W(mask[0])
        tgt_H, tgt_W = self.size

        if X == 0 and Y == 0 and H == image.shape[1] and W == image.shape[2]: # unsegmentable
            Xn = torch.randint(0, H - tgt_H, (), generator=self.generator)
            Yn = torch.randint(0, H - tgt_W, (), generator=self.generator)
        else:
            shift_H = torch.randint(0, tgt_H - H, (), generator=self.generator) if tgt_H > H else 0
            shift_W = torch.randint(0, tgt_W - W, (), generator=self.generator) if tgt_W > W else 0
            Xn = max(min(X - shift_H, image.shape[1] - tgt_H), 0)
            Yn = max(min(Y - shift_W, image.shape[2] - tgt_W), 0)

        Yx = Yn + tgt_W
        Xx = Xn + tgt_H

        if Yn < 0 or Xn < 0 or Yx > image.shape[2] or Xx > image.shape[1]:
            raise ValueError(f"Crop bounds {Xn}:{Xx}, {Yn}:{Yx} are out of bounds for image of shape {image.shape}")

        return image[:, Xn:Xx, Yn:Yx], mask[:, Xn:Xx, Yn:Yx]


class SequentialTwoInputs(nn.Module):
    """A sequential container for modules that take two inputs and return two outputs.

    Similar to nn.Sequential but designed for transformations that operate on pairs
    of inputs (like image and mask).
    """

    def __init__(self, *args):
        """
        Args:
            *args: Modules to be added to the container in the order they should be applied.
        """
        super(SequentialTwoInputs, self).__init__()
        self.transforms = nn.ModuleList(args)

    def forward(self, x1, x2):
        """
        Args:
            x1: First input (e.g., image)
            x2: Second input (e.g., mask)

        Returns:
            tuple: Transformed versions of both inputs
        """
        for transform in self.transforms:
            x1, x2 = transform(x1, x2)
        return x1, x2


def compute_densities(X):
    # X: (n_samples, n_features)

    # Whiten our samples
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Whitening matrix
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    X_white = X_centered @ W

    # Compute bandwidth as Scott's factor
    n, d = X.shape
    bw = np.power(n, -1.0 / (d + 4))

    # Fit KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(X_white)

    # Compute sample densities in batches
    b = 1024
    densities = list()
    for i in range(0, X_white.shape[0], b):
        densities.append(np.exp(kde.score_samples(X_white[i:i+b])).get())
    densities = np.concatenate(densities)

    return densities


class LMDataset(Dataset):
    """

    Properties:
        data (Tensor)                          : images
        sample_labels (list of Tensors)        : labels for each sample
        paths (tuple)                          : original path of each image
        transform (Transform)                  : transform to apply before returning images
        label (list)                           : a list of one or more labels to use for supervised learning

    """

    FC_COLS = [
      'FSC-A', 'FSC-H', 'FSC-W',
      'SSC-A', 'SSC-H', 'SSC-W',
      'BL1-A', 'BL1-H', 'BL1-W',
      'BL2-A', 'BL2-H', 'BL2-W',
      'BL3-A', 'BL3-H', 'BL3-W',
      'YL1-A', 'YL1-H', 'YL1-W',
      'YL2-A', 'YL2-H', 'YL2-W',
      'YL3-A', 'YL3-H', 'YL3-W',
      'YL4-A', 'YL4-H', 'YL4-W'
    ]

    FC_NORM = {
            'FSC-A': 1048576.0,
            'FSC-H': 1048576.0,
            'FSC-W': 1048576.0,
            'SSC-A': 1048576.0,
            'SSC-H': 1048576.0,
            'SSC-W': 1048576.0,
    }


    __regression_labels = {'time'}

    split_vals = {'train': 0, 'validation': 1, 'test': 2}

    @classmethod
    def is_regression(cls, label):
        return (label in cls.__regression_labels) or (label in cls.FC_COLS)

    def __init__(self, path, label_classes=None, return_labels=False, logger=None, transform=None,
                 label=None, split=None, rand_split=False, exp_split=False, split_seed=None, return_weights=False):
        """

        Args:
            path (str)                  : path to the HDMF DynamicTable file
            label_classes (dict)        : a dictionary of classes for each label
            images (bool)               : whether or not to return images
            use_return_labels (bool)    : whether or not to return labels for each data point
            logger (Logger)             : the logger to use when loading data
            transform (Transform)       : the transform to apply to each image
            label (str, list, tuple)    : the label(s) to return when getting a sample
            split (str)                 : a string indication which split to use ("train" or "val")
            exp_split (bool)            : calculate train-validation-test split using metdata on campaign, condition,
                                          and ht (a.k.a. replicate). validation and test split are first calculated
                                          by getting single unique ht for each condition for each campaign. This is
                                          then split between campaigns. All remaining samples are left for training.
                                          exp_split is short for "experimental split"

        """
        # When exp_split == False and rand_split == False, no there is not split.
        if rand_split and exp_split:
            raise ValueError("rand_split and exp_split cannot both be True")

        self.logger = logger or get_logger('warning')

        self.path = path

        self.return_labels = return_labels

        if return_labels:
            if not isinstance(label, (tuple, list)) or len(label) == 1:
                label = label[0] if isinstance(label, (tuple, list)) else label
                if label.lower() == 'fcs':
                    label = self.FC_COLS
                else:
                    label = [label]

        # The labels to return
        self.label = label

        # Labels for each sample
        self.sample_labels = None
        self.weights = None

        self.transform = transform

        # The split to use. Should be 'train', 'validation', or 'test'
        self.split = split

        # The mask to use for selecting the samples that should be returned.
        # Should take on the values fonud in self.split_vals
        self.split_mask = None

        # The actual indices of the data subset to return
        self.subset = None

        self.io = None
        self.table = None

        self.open()

        if return_weights:
            X = np.array(self.sample_labels).T
            densities = compute_densities(X)
            self.weights = self._compute_weights(densities).astype(np.float32)

        self.__len = len(self.table)

        self.exp_split = exp_split
        self.rand_split = rand_split

        if exp_split:
            # Use this function if you ever want to revisit campaign-based splitting
            # train_mask, val_mask, test_mask = make_masks(self.table, split_validation=True, seed=split_seed)

            # For now, lets just split based on ht
            train_mask, val_mask, test_mask = make_ht_masks(self.table, seed=split_seed)
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

    def _compute_weights_log(self, densities):
        weights = -np.log10(densities/densities.sum())
        a, b = weights.min(), weights.max()
        weights = (weights - a) / (b - a)
        return weights

    def _compute_weights(self, densities):
        weights = 1 / densities
        a, b = weights.min(), weights.max()
        weights = (weights - a) / (b - a)
        return weights

    def open(self, worker_id=None):
        """Open file for reading if it is not currently open"""
        if self.table is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="EnumData is experimental.*", category=UserWarning)
                self.io = get_hdf5io(self.path, 'r')
                self.table = self.io.read()


            self.label_classes = dict()

            if self.return_labels:
                self.sample_labels = []
                for k in self.label:
                    if isinstance(self.table[k], EnumData):
                        self.label_classes[k] = self.table[k].elements.data[:]
                    else:
                        self.label_classes[k] = None

                    # Convert to 32-bit floating point to avoid issues with Torch
                    # (By default, Torch assumes single precision floats)
                    labels = self.table[k].data[:]
                    if isinstance(labels[0], np.floating):
                        labels = labels.astype(np.float32)
                    self.sample_labels.append(labels)


    def close(self):
        self.io.close()
        self.table = None
        self.io = None

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

        image = torch.as_tensor(self.table['images'].data[i]).unsqueeze(0)
        mask = torch.as_tensor(self.table['masks'].data[i]).unsqueeze(0)

        ret = image
        if self.transform is not None:
            # TODO: self.transform should account for an image or an image plus mask.
            # This is so we can do transformations with the segmentation mask
            ret = self.transform(image, mask)
            if isinstance(self.transform, SequentialTwoInputs):
                ret = ret[0]

        if self.sample_labels is None:
            return ret
        else:
            ret_tmp = [torch.as_tensor(self.sample_labels[j][i])
                       for j in range(len(self.sample_labels))]

            # Keep this here, since it's for mulltlabel loss when we have different loss types
            # return ret, tuple(ret_tmp)

            if self.weights is not None:
                return ret, torch.Tensor(ret_tmp), torch.tensor(self.weights[i])
            else:
                return ret, torch.Tensor(ret_tmp)

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


TRANSFORMS = {
        'blur': GaussianBlur(3, sigma=(0.01, 1.0)),
        'rotate': RandomRotation(180),
        'random_crop': RandomCrop((64, 64)),
        'center_crop': T.CenterCrop((64, 64)),
        'hflip': RandomHorizontalFlip(0.5),
        'vflip': RandomVerticalFlip(0.5),
        'noise': GaussianNoise(sigma=(10, 12)),
        'rgb': RGB(),
        'float': Float(),
        'norm': Norm(),
        'masknorm': MaskNorm(),
}


def get_transforms(*transforms):
    """Return a transforms appropriate for Ambr light microscopy data

    The following transforms and their respective keys are:
        norm:           Normalize the image by subtracting the mean pixel value from
                        image
        blur:           A Gaussian Blur
        rotate:         Random rotation of up to 180 degrees in either direction
        random_crop:    Random crop around center to 64x64 pixels
        center_crop:    Center crop images to 64x64 pixels
        hflip:          Random horizontal rotation with probability of 0.5
        vflip:          Random vertical rotation with probability of 0.5
        noise:          Randomly apply noise with probability of 0.5. Noise magnitude
                        will be between 10-12 signal-to-noise ratio
        rgb:            Convert to RGB format i.e. 3 channels
        float:          Convert to torch.Float dtype

    Args:
        transforms: the list of transforms to get. Valid options are 'blur', 'rotate', 'random_crop',
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
    return ret[0] if len(ret) == 1 else SequentialTwoInputs(*ret)


def make_ht_masks(table, seed=None):
    """
    Split images into train, validation, test based on ht
    """
    ht = table['ht'].data[:]
    uniq_ht = np.unique(ht)

    np.random.default_rng(seed).shuffle(uniq_ht)

    test_ht = uniq_ht[0]
    val_ht = uniq_ht[1]

    train_mask = np.ones(len(ht), dtype=bool)

    test_mask = ht == test_ht
    val_mask = ht == val_ht

    train_mask[test_mask] = False
    train_mask[val_mask] = False

    assert (train_mask.sum() + val_mask.sum() + test_mask.sum()) == len(table)

    return train_mask, val_mask, test_mask




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
    if 'source' in table:
        val_sample = np.where(table['source'].elements.data[:] == 'sample')[0][0]
        mask = table['source'].data[:] == val_sample
    else:
        mask = np.s_[:]

    if 'campaign' in table:
        campaign = table['campaign'].data[:][mask]
    else:
        campaign = np.array(['fake_campaign'] * (len(table) if isinstance(mask, slice) else mask.sum()))

    # Get unique campaigns and potentially split them
    unique_campaigns = np.sort(np.unique(campaign))
    if split_validation and len(unique_campaigns) == 1:
        msg = ("Cannot split based on campaigns, since there is only a single campaign present.\n"
               "You must include a 'campaign' column with two or more campaigns to split based on experimental metadata")
        raise ValueError(msg)

    condition = table['condition'].data[:][mask]
    ht = table['ht'].data[:][mask]

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
