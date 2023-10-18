import logging
import sys


def parse_logger(string, stream=sys.stderr):
    if not string:
        ret = logging.getLogger()
        hdlr = logging.StreamHandler(stream)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret


def get_logger():
    return parse_logger('')
