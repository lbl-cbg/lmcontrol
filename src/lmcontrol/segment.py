import argparse
import glob
import os

from numba import njit
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import median
import skimage.io as sio
from skimage.measure import label
from skimage.morphology import closing
from tqdm import tqdm

from .utils import get_logger


@njit
def _raw_outlier_mask(img):
    mean = img.ravel().mean()
    std = img.ravel().std()
    mask = np.logical_or(img < mean - std, img > mean + std).astype(np.int64)
    mask[:15] = 0
    mask[-15:] = 0
    mask[:, :15] = 0
    mask[:, -15:] = 0
    return mask


def outlier_threshold_tol(img, tol=0.001, max_med=10):
    """
    Threshold gray scale image keeping only those pixels on the tails of the
    pixel value distribution.

    Args:
        tol (float)      : the tolerance for applying successive median filters.
                           If the change is less than this, stop applying filters
        max_med (int)    : the maximum number of median filters to apply
        ntrim (int)      : the size of the border to ignore
        nsig (int)       : the number of standard deviations above/below which to
                           set the threshold
    """

    mask = _raw_outlier_mask(img)

    for i in range(max_med):
        tmp = median(mask.astype(int))
        diff = (mask != tmp).mean()
        if diff < tol:
            break
        mask = tmp

    return mask


class UnsegmentableError(ValueError):
    """An Exception to raise when segmentation fails for known reasons"""
    pass


def outlier_cluster(img, frac=0.3, tol=0.001, max_med=10):
    """
    Find mask for cell using outlier thresholding.

    Args:
        tol (float)      : the tolerance for applying successive median filters.
                           If the change is less than this, stop applying filters
        max_med (int)    : the maximum number of median filters to apply
        ntrim (int)      : the size of the border to ignore
        nsig (int)       : the number of standard deviations above/below which to
                           keep set the threshold
        frac (float)     : the minimum fraction of pixels required to belong to the
                           main cluster for declaring a main cluster found
    """
    mask = outlier_threshold_tol(img)
    mask = closing(mask)
    if mask.sum() / mask.size < 0.005:
        raise UnsegmentableError("Not enough masked pixels to to cluster")
    if frac is not None:
        labels = label(mask)
        counts = np.bincount(labels.ravel())[1:]
        counts = (counts / counts.sum())
        cell_cluster_id = np.where(counts >= frac)[0]
        if len(cell_cluster_id) == 0:
            raise UnsegmentableError("Unable to find one majority cluster")

        cell_cluster_id = cell_cluster_id[0] + 1
        mask = (labels == cell_cluster_id).astype(int)
    return mask


def _adjust_bounds(Xn, Xx, Yn, Yx, target_size):
        Ydim = Yx - Yn
        Xdim = Xx - Xn
        Xpad = (target_size[0] - Xdim) // 2
        Ypad = (target_size[1] - Ydim) // 2
        Xx += Xpad
        Xn -= Xpad
        Yx += Ypad
        Yn -= Ypad
        if (Xx - Xn) != target_size[0]:
            Xx += 1
        if (Yx - Yn) != target_size[1]:
            Yx += 1
        return Xn, Xx, Yn, Yx


def trim_box(mask, img, size=None, pad=True):
    """
    Crop image using bounding box calculated using the provided mask. The
    mask should have a single cluster for this function to have the intended
    effect.

    Args:
        mask (array)        : the mask to use for cropping the image
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that
                              *img* should be cropped to
        pad (bool)          : If True, pad with zeros around image to shape specified by *size*,
                              otherwise crop image to the specified size, centered on the mask.

    """
    mask = mask.astype(int)
    Y, X = np.where(mask == 1)
    Yx, Yn = Y.max() + 1, Y.min()
    Xx, Xn = X.max() + 1, X.min()

    if size is not None:
        if not pad:
            Yn, Yx, Xn, Xx = _adjust_bounds(Yn, Yx, Xn, Xx, size)
            ret = img[Yn: Yx, Xn: Xx]
        else:
            ret = img[Yn: Yx, Xn: Xx]
            image_height, image_width = ret.shape
            crop_height, crop_width = size
            padding_ltrb = [[0, 0], [0, 0]]
            w_d = crop_width - image_width
            if w_d > 0:
                padding_ltrb[1][0] = w_d // 2
                padding_ltrb[1][1] = w_d // 2 + int(w_d % 2)
            h_d = crop_height - image_height
            if h_d > 0:
                padding_ltrb[0][0] = h_d // 2
                padding_ltrb[0][1] = h_d // 2 + int(h_d % 2)
            ret = np.pad(ret, padding_ltrb, mode='constant', constant_values=0)
            Yn, Yx, Xn, Xx = _adjust_bounds(0, ret.shape[0], 0, ret.shape[1], size)
            ret = ret[Yn: Yx, Xn: Xx]
    else:
        ret = img[Yn: Yx, Xn: Xx]

    if 0 in ret.shape:
        raise ValueError(f"img has zero area, shape is {ret.shape}")
    return ret


def crop_image(img, size=None):
    """
    Crop image around the cell imaged in *img*.

    Args:
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that

    """
    mask = outlier_cluster(img)
    new_img = trim_box(mask, img, size=size)
    return new_img


def metadata(string):
    if len(string):
        try:
            ret = {k:v for k, v in (tuple(kv.split("=")) for kv in string.split(","))}
        except:
            raise argparse.ArgumentTypeError()
        return ret
    return dict()


def crop_images(argv=None):
    """
    Segment and images in a directory, saving segmented images to a
    new directory. This will also save all images cropped and stored in
    an ndarray for easy loading of cropped data.
    """

    def crop_size(string):
        if len(string) == 0:
            return (64, 32)
        else:
            try:
                x, y = string.split(',')
                x, y = int(x), int(y)
                return (x, y)
            except:
                raise ArgumentTypeError()

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help='The directory containing images to crop')
    parser.add_argument("output_dir", type=str, help='The directory to save cropped images to')
    parser.add_argument("-u", "--save-unseg", action='store_true', default=False,
                        help="Save unsegmentable images in output_dir under directory 'unseg'")
    parser.add_argument('-c', '--crop', type=crop_size, default=(64, 32), metavar='SHAPE',
                        help='the size to crop images to (in pixels) for saving as ndarray. default is (64, 32)')
    parser.add_argument('-p', '--pad', default=False, action='store_true',
                        help='pad segmented image with zeros to size indicated with --crop. Otherwise use pad with original image contents')
    parser.add_argument("-m", "--metadata", help="a comma-separated list of key=value pairs. e.g. ht=1,time=S4", default="", type=metadata)
    args = parser.parse_args(argv)

    logger = get_logger()

    dir_exists = False # only create unseg directory if there were any unsegmentable images
    unseg_dir = os.path.join(args.output_dir, 'unseg')

    seg_images = list()
    seg_masks = list()
    paths = list()
    orig_seg_images = list()
    n_unseg = 0

    image_paths = glob.glob(os.path.join(args.image_dir, "*.tif"))

    logger.info(f"Found {len(image_paths)} images in {args.image_dir}")
    logger.info(f"Saving segmented images to {args.output_dir}")
    if args.save_unseg:
        logger.info(f"Saving any unsegmented images to {unseg_dir}")
    else:
        logger.info("Discarding unsegmented images")

    npz_out = os.path.join(args.output_dir, "all_processed.npz")
    logger.info(f"Cropping images to {args.crop} and saving to {npz_out} for convenience")

    for tif in tqdm(image_paths):
        image = sio.imread(tif)[:, :, 0]
        try:
            #check for bad images caused by flow-cytometer and/or imaging errors
            if image.std() > 15:
                raise UnsegmentableError("StdDev of pixels is quite high, this is probably a bad image")
            mask = outlier_cluster(image)
            segi = trim_box(mask, image)
            orig_seg_images.append(segi) # save segmented original image
            paths.append(tif)

            # rotate image if cell is oriented horizontally
            # and crop to standard size
            if (segi.shape[1] / segi.shape[0]) >= 1.4:
                image = ndi.rotate(image, -90)
                mask = ndi.rotate(mask, -90)
            segi = trim_box(mask, image, size=args.crop, pad=args.pad)
            segm = trim_box(mask, mask, size=args.crop, pad=args.pad)

            seg_images.append(segi)
            seg_masks.append(segm)
        except UnsegmentableError:
            if args.save_unseg:
                target = os.path.join(unseg_dir, os.path.basename(tif))
                if not dir_exists:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    dir_exists = True
                sio.imsave(target, image)
            n_unseg += 1
            continue

    logger.info(f"Done segmenting images. {n_unseg} ({100 * n_unseg / len(image_paths):.1f}%) images were unsegmentable")

    seg_images = np.array(seg_images)
    seg_masks = np.array(seg_masks)
    paths = np.array(paths)

    # Save segmented original images
    os.makedirs(args.output_dir, exist_ok=True)
    for image, path in zip(orig_seg_images, paths):
        target = os.path.join(args.output_dir, os.path.basename(path))
        sio.imsave(target, image)

    # save cropped and rotate
    logger.info(f"Saving all cropped images to {npz_out}")
    np.savez(npz_out, masks=seg_masks, images=seg_images, paths=paths, **args.metadata)
