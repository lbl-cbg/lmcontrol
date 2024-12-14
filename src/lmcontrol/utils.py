import logging
import sys

from pkg_resources import resource_filename
import yaml

def parse_logger(string, stream=sys.stderr, level='info'):
    if not string:
        ret = logging.getLogger()
        hdlr = logging.StreamHandler(stream)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)

    ret.setLevel(getattr(logging, level.upper()))
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    return ret


def get_logger(level='info'):
    return parse_logger('', level=level)


def get_metadata_info():
    file_path = resource_filename(__package__, 'metadata_info.yaml')
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
