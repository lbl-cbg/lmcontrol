from .dataset import LMDataset
from ..utils import get_logger


def get_loaders(args, train_tfm=None, val_tfm=None, return_labels=True, logger=None):
    logger = logger or get_logger('warning')

    tdset_kwargs = dict(n_samples=args.n_samples, logger=logger, return_labels=return_labels)
    for attr in ('label', ):
        if hasattr(args, attr):
            tdset_kwargs[attr] = getattr(args, attr)
    if args.val_frac:
        train_files = args.training
        val_files = args.training
        tdset_kwargs.update(dict(split='train', val_size=args.val_frac, seed=args.seed))
        vdset_kwargs = tdset_kwargs.copy()
        vdset_kwargs['split'] = 'validate'
    elif args.validation:
        train_files = args.training
        val_files = args.validation
        vdset_kwargs = tdset_kwargs.copy()
    else:
        print("You must specify --validation or --val_frac", file=sys.stderr)
        exit(1)

    logger.info(f"Loading training data: {len(train_files)} files")
    train_dataset = LMDataset(train_files, transform=train_tfm, **tdset_kwargs)

    logger.info(f"Loading validation data: {len(val_files)} files")
    val_dataset = LMDataset(val_files, transform=val_tfm, label_classes=train_dataset.label_classes, **vdset_kwargs)

    dl_kwargs = dict(num_workers=0 if args.debug else 2, batch_size=args.batch_size)

    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    return train_loader, val_loader
