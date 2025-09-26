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
        'Data preparation': {
            'segment': Command('segment.main', 'Segment light microscopy images to bounding box around single cell'),
            'merge': Command('merge.main', 'Merge data into monolithic HDMF file'),
            'add-md': Command('segment.add_metadata', 'Add metadata to cropped image package'),
            'make-commands': Command('command.main', 'Make commands for cropping'),
            'ls-metadata': Command('merge.ls_metadata', 'List the metadata available in segmentation NPZ'),
        },
        'Visualization': {
            'emb-viz': Command('viz.interactive_viz.main', 'Interactive viewer of image embeddings'),
        },
        'Self-supervised Learning': {
            'train-byol': Command('nn.byol.train', 'Train BYOL model'),
            'infer-byol': Command('nn.byol.predict', 'Run with inference BYOL backbone'),
        },
        'Supervised Learning':{
            'tune': Command('nn.clf.tune', 'Run HPO for classifier or regressor with Optuna'),
            'sup-train': Command('nn.clf.train', 'Train classifier or regressor to predict bioreactor properties from images'),
            'sup-predict': Command('nn.clf.predict', 'Predict bioreactor properties from images using trained model')
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
            argv = sys.argv[2:]
            sys.argv[0] = sys.argv[0] + " " + sys.argv[1]
            func(argv)
        else:
            print("Unrecognized command: '%s'" % cmd, file=sys.stderr)

