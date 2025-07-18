import argparse

from pkg_resources import resource_filename

from .utils import get_logger, get_metadata_info
from .data_utils import load_npzs

from hdmf.spec import GroupSpec, DatasetSpec, NamespaceBuilder
from hdmf.backends.hdf5 import HDF5IO, H5DataIO
from hdmf.common import DynamicTable, load_namespaces, get_hdf5io
import numpy as np
from sklearn.preprocessing import LabelEncoder

def _nspath():
    return resource_filename(__package__, 'namespace.yaml')

def make_spec(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--namespace_path", help="where to save the namespace file", default=None)
    parser.add_argument("-s", "--spec_path", help="where to save the spec file", default=None)
    args = parser.parse_args(argv)

    if args.namespace_path is None:
        args.namespace_path = _nspath()
    if args.spec_path is None:
        args.spec_path = 'single_cell_images.yaml'

    sci_spec = GroupSpec(
        doc='A DynamicTable without predefined columns',
        data_type_inc='DynamicTable',
        data_type_def='SingleCellImages',
        name='metadata',
        datasets=[
            DatasetSpec(
                doc='Unprocessed single cell images',
                dtype='uint8',
                shape=(None, None, None),
                name='raw_images',
                data_type_inc='VectorData',
                quantity='?',
            ),
            DatasetSpec(
                doc='Cropped single cell images',
                dtype='uint8',
                shape=(None, None, None),
                name='images',
                data_type_inc='VectorData'
            ),
            DatasetSpec(
                doc='The segmentation masks for each image',
                dtype='uint8',
                shape=(None, None, None),
                name='masks',
                data_type_inc='VectorData'
            ),
            DatasetSpec(
                doc='A dataset for storing images with a single channel',
                dtype='text',
                shape=(None,),
                name='paths',
                data_type_inc='VectorData'
            ),
        ]
    )

    # Create the namespace for the specifications
    nsb = NamespaceBuilder(
        doc='A namespace for storing SingleCellImages',
        name='sci_namespace',
        version='1.0.0',
    )

    nsb.include_namespace('hdmf-common')
    nsb.add_spec(args.spec_path, sci_spec)
    nsb.export(args.namespace_path)


def ls_metadata(argv=None):
    """
    Prints the keys and values inside multiple NPZ files.
    If the value is a scalar or a single-element ndarray, it prints the value.
    If the value is an ndarray with more than one element, it prints the shape and data type.

    Parameters:
    argv (list): Command line arguments (default is None).
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Print keys and values from multiple NPZ files.')
    parser.add_argument('file_paths', type=str, nargs='+', help='The paths to the NPZ files.')

    # Parse the command-line arguments
    args = parser.parse_args(argv)

    # Iterate over each file path provided
    for file_path in args.file_paths:
        print(f"\n{file_path}")
        try:
            with np.load(file_path) as data:
                for key in data.keys():
                    value = data[key]
                    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
                        print(f"{key:<16}: {value.item()}")  # Use .item() to get the scalar value
                    else:
                        print(f"{key:<16}: shape={value.shape}, dtype={value.dtype}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")


def main(argv=None):

    desc = "Merge cropped images into a single HDF5 file"
    epi = """
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("output_h5", help="the path of the HDF5 to write images to")
    parser.add_argument("npzs", nargs="+", help="the NPZ files with cropped images to merge")
    parser.add_argument("-f", "--force", action='store_true', default=False, help="ignore checks, just write my file")
    parser.add_argument("-c", "--compress", action='store_true', default=False, help="compress images and masks")

    args = parser.parse_args(argv)

    logger = get_logger()

    if not args.output_h5.endswith('.h5') and not args.force:
        logger.critical(f"output file {args.output_h5} does not end with .h5 - Naively failing to prevent overwriting your data")
        exit(1)

    metadata_info = get_metadata_info()

    masks, images, paths, metadata = load_npzs(args.npzs, logger=logger)

    table = DynamicTable(name='single_cell_images',
                         description='Single cell images from ABPDU',
                         id=np.arange(len(masks)))

    if args.compress:
        images = H5DataIO(images, compression='gzip', chunks=(100, 152, 152))
        masks = H5DataIO(masks.astype('uint8'), compression='gzip')
    else:
        masks = masks.astype('uint8')
    table.add_column('images', 'Cropped single cell images', data=images)
    table.add_column('masks', 'The segmentation masks for each image', data=masks)
    table.add_column('paths', 'the original file path for the raw image', data=paths)
    missing_desc = list()
    for key, value in metadata.items():
        if key not in metadata:
            logger.warning(f"Skipping key '{key}' - Not found in metadata info")
            continue

        if key not in metadata_info:
            missing_desc.append(key)
            continue
        if metadata_info[key]['enum']:
            enc = LabelEncoder().fit(value)
            value = enc.transform(value).astype(np.uint64)
            table.add_column(key, metadata_info[key]['description'], data=value, enum=enc.classes_)
        else:
            table.add_column(key, metadata_info[key]['description'], data=value.astype(float))

    if len(missing_desc) > 0:
        missing_desc = ", ".join(missing_desc)
        if args.force:
            logger.info(f"Ignoring the following columns due to missing metadata: {missing_desc}")
        else:
            msg = f"The following keys do not have metadata info. Use --force flag to ignore them, or add their metadata info src/lmcontrol/metadata_info.yaml\n{missing_desc}"
            logger.info(msg)
            exit(1)

    logger.info(f"Writing to {args.output_h5}")
    with get_hdf5io(args.output_h5, 'w') as io:
        io.write(table)

    logger.info(f"DONE")

if __name__ == "__main__":
    main()
