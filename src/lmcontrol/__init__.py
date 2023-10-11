import numpy as np
from importlib import import_module

class Command:
    def __init__(self, module, doc):
        ar = f'lmcontrol.{module}'.split('.')
        self.pkg = '.'.join(ar[:-1])
        self.func = ar[-1]
        self.doc = doc

    def get_func(self):
        return getattr(import_module(self.pkg), self.func)


def main():

    command_dict = {
        'Segmentation': {
            'crop': Command('segment.crop', 'Crop light microscopy images to bounding box around single cell'),
        },
    }
    import sys
    if len(sys.argv) == 1:
        print('Usage: lmcontrol <command> [options]')
        print('Available commands are:\n')
        for g, d in command_dict.items():
            print(f' {g}')
            for c, f in d.items():
                nspaces = 16 - len(c)
                desc = ''
                print(f'    {c}' + ' '*nspaces + f.doc)
            print()
    else:
        cmd = sys.argv[1]
        for g, d in command_dict.items():
            func = d.get(cmd)
            if func is not None:
                func = func.get_func()
                break
        if func is not None:
            func(sys.argv[2:])
        else:
            print("Unrecognized command: '%s'" % cmd, file=sys.stderr)

