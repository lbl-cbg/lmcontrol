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
            'crop': Command('segment.main', 'Crop light microscopy images to bounding box around single cell'),
            'prep-viz': Command('viz.make_package.main', 'Prepare package for visualizing image embeddings'),
            'emb-viz': Command('viz.interactive_viz.main', 'Interactive viewer of image embeddings'),
        },
        'Self-supervised Learning': {
            'train-byol': Command('nn.byol.train', 'Train BYOL model'),
            'infer-byol': Command('nn.byol.predict', 'Run with inference BYOL backbone'),
            
        },
        'Supervised Learning':{
            'train-clf': Command('nn.clf.train', 'Train with classifier'),
            'predict-clf': Command( 'nn.clf.predict', 'Predict with classifier'),
            'stratify-clf': Command('nn.stratify-clf.stratify', 'Training and predicting with train_test_split stuff'),
            'train-clf-time-reactor': Command('nn.main_clf_time_reactor.train', 'clf just wth equal data points for reactors too'), 
            'predict-clf-time-reactor': Command('nn.main_clf_time_reactor.predict', 'clf just wth equal data points for reactors too'), 
            'misclassify-clf-time-reactor': Command('nn.clf_time_reactor.misclassify', 'stores misclassified points just wth equal data points for reactors too'),
            'train-superset': Command('nn.superset_clf_stratify_time.train', 'Superset train with classifier'),
            'predict-superset': Command('nn.superset_clf_stratify_time.predict', 'Superset predict with classifier'),
            'predict-confusion': Command('nn.superset_clf_stratify_time.predict', 'confusion_labels STUFF with Superset predict as classifier'),
            'train-tr-val-clf-time-reactor': Command('nn.tr_val_clf_time_reactor.train', 'Tr=0.8,val=0.2 and clf used'),
            'predict-tr-val-clf-time-reactor': Command('nn.tr_val_clf_time_reactor.predict', 'Tr=0.8,val=0.2 and clf used'),
            'opta-args-train-main-clf-time-reactor': Command('nn.optatune_main_clf_time_reactor._add_training_args', 'Optatune used code training_args'),
            'opta-train-main-clf-time-reactor': Command('nn.optatune_main_clf_time_reactor.train', 'Optatune used code for training'),
            'opta-perdict-main-clf-time-reactor': Command('nn.optatune_main_clf_time_reactor.predict', 'Optatune used code for predicting'),
            'opta-tune': Command('nn.optatune_main_clf_time_reactor.tune', 'Optatune used code for tune function'),
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
            argv = sys.argv[2:]
            sys.argv[0] = sys.argv[0] + " " + sys.argv[1]
            func(argv)
        else:
            print("Unrecognized command: '%s'" % cmd, file=sys.stderr)

