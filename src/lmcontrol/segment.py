import argparse
import glob
import os

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import median
import skimage.io as sio
from skimage.measure import label
from skimage.morphology import closing
from tqdm import tqdm


def outlier_threshold_tol(img, tol=0.001, max_med=10, ntrim=15, nsig=1):
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
    mean = img.ravel().mean()
    std = img.ravel().std()
    mask = np.logical_or(img < mean - nsig*std, img > mean + nsig*std).astype(int)
    if ntrim > 0:
        mask[:ntrim] = 0
        mask[-ntrim:] = 0
        mask[:, :ntrim] = 0
        mask[:, -ntrim:] = 0

    for i in range(max_med):
        tmp = median(mask.astype(int))
        diff = (mask != tmp).mean()
        if diff < tol:
            break
        mask = tmp

    return mask


def outlier_cluster(img, frac=0.3, tol=0.001, max_med=10, ntrim=15, nsig=1):
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
    mask = outlier_threshold_tol(img, ntrim=15)
    mask = closing(mask)
    if frac is not None:
        labels = label(mask)
        counts = np.bincount(labels.ravel())[1:]
        counts = (counts / counts.sum())
        cell_cluster_id = np.where(counts >= frac)[0]
        if len(cell_cluster_id) == 0:
            raise ValueError("Unable to find one majority cluster")

        cell_cluster_id = cell_cluster_id[0] + 1
        mask = (labels == cell_cluster_id).astype(int)
    return mask


def trim_box(mask, img, size=None):
    """
    Crop image using bounding box calculated using the provided mask. The
    mask should have a single cluster for this function to have the intended
    effect.

    Args:
        mask (array)        : the mask to use for cropping the image
        img  (array)        : the image to crop
        size (tuple, list)  : the target size around the cluster found in *mask* that
                              *img* should be cropped to

    """
    mask = mask.astype(int)
    X, Y = np.where(mask == 1)
    Xx, Xn = X.max(), X.min()
    Yx, Yn = Y.max(), Y.min()
    if size is not None:
        Ydim = Yx - Yn
        Xdim = Xx - Xn
        Xpad = (size[0] - Xdim) // 2
        Ypad = (size[1] - Ydim) // 2
        Xx += Xpad
        Xn -= Xpad
        Yx += Ypad
        Yn -= Ypad
        if (Xx - Xn) != size[0]:
            Xx += 1
        if (Yx - Yn) != size[1]:
            Yx += 1
    ret = img[Xn: Xx, Yn: Yx]
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
    args = parser.parse_args(argv)


    dir_exists = False # only create unseg directory if there were any unsegmentable images

    seg_images = list()
    seg_masks = list()
    paths = list()
    orig_seg_images = list()
    n_unseg = 0
    for tif in tqdm(glob.glob(os.path.join(args.image_dir, "*.tif"))):
        image = sio.imread(tif)[:, :, 0]
        try:
            #TODO: Add check for garbage images i.e. overdispersed images.
            if image.std() > 15:
                raise ValueError("StdDev of pixels is quite high, this is probably a bad image")
            mask = outlier_cluster(image)
            segi = trim_box(mask, image)
            orig_seg_images.append(segi) # save segmented original image

            # rotate image if cell is oriented horizontally
            # and crop to standard size
            if (segi.shape[1] / segi.shape[0]) >= 1.4:
                image = ndi.rotate(image, -90)
                mask = ndi.rotate(mask, -90)
            segi = trim_box(mask, image, size=args.crop)
            segm = trim_box(mask, mask, size=args.crop)

            seg_images.append(segi)
            seg_masks.append(segm)
            paths.append(tif)
        except ValueError:
            if args.save_unseg:
                target = os.path.join(args.output_dir, 'unseg', os.path.basename(tif))
                if not dir_exists:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    dir_exists = True
                sio.imsave(target, img)
            n_unseg += 1
            continue

    seg_images = np.array(seg_images)
    seg_masks = np.array(seg_masks)
    paths = np.array(paths)

    # Save segmented original images
    os.makedirs(args.output_dir, exist_ok=True)
    for image, path in zip(orig_seg_images, paths):
        target = os.path.join(args.output_dir, os.path.basename(path))
        sio.imsave(target, image)

    # save cropped and rotate
    np.savez(os.path.join(args.output_dir, "all_processed.npz"), masks=seg_masks, images=seg_images, paths=paths)
