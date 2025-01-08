import argparse

from pkg_resources import resource_filename

from .utils import get_logger, get_metadata_info
from .data_utils import load_npzs

from hdmf.spec import GroupSpec, DatasetSpec, NamespaceBuilder
from hdmf.backends.hdf5 import HDF5IO
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


def main(argv=None):

    desc = "Merge cropped images into a single HDF5 file"
    epi = """
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi)
    parser.add_argument("output_h5", help="the path of the HDF5 to write images to")
    parser.add_argument("npzs", nargs="+", help="the NPZ files with cropped images to merge")
    parser.add_argument("-n", "--namespace_path", help="The namespace specification file to use", default=None)
    parser.add_argument("-f", "--force", action='store_true', default=False, help="ignore checks, just write my file")

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

    table.add_column('images', 'Cropped single cell images', data=images.astype('uint8'))
    table.add_column('masks', 'The segmentation masks for each image', data=masks.astype('uint8'))
    table.add_column('paths', 'the original file path for the raw image', data=paths)
    for key, value in metadata.items():
        if metadata_info[key]['enum']:
            enc = LabelEncoder().fit(value)
            value = enc.transform(value).astype(np.uint64)
            table.add_column(key, metadata_info[key]['description'], data=value, enum=enc.classes_)
        else:
            table.add_column(key, metadata_info[key]['description'], data=value.astype(float))

    with get_hdf5io(args.output_h5, 'w') as io:
        io.write(table)


if __name__ == "__main__":
    main()
