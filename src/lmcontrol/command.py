import argparse
from collections import Counter
import os
import sys

import pandas as pd


def spreadsheet(s):
    sheet = 0
    f = s
    if ':' in f:
        f, tmp = f.split(':')
        try:
            sheet = int(tmp)
        except Exception:
            sheet = tmp

    if f.endswith('xlsx'):
        df = pd.read_excel(f, sheet_name=sheet, header=0)
    else:
        df = pd.read_csv(f, header=0)

    return df


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('campaign', help='The campaign name')
    parser.add_argument('data_dir', help='The campaign name')
    parser.add_argument('ht_metadata', type=spreadsheet, help='a spreadsheet with metadata about HTs')
    parser.add_argument('sample_metadata', type=spreadsheet, help='a spreadsheet with metadata about samples')
    parser.add_argument('-o', '--outdir', help='The output directory command should write to', default='$OUTDIR')
    parser.add_argument('-a', '--arguments', help='Additional arguments to lmcontrol crop', default='-n -c 96,96 -C -u')

    args = parser.parse_args(argv)

    ht_metadata = args.ht_metadata
    sample_metadata = args.sample_metadata

    all_samples = dict()
    for i, sample in sample_metadata.iterrows():
        for j, ht in ht_metadata.iterrows():
            md = dict(
                    campaign=args.campaign,
                    time=f"{sample['Time']:0.1f}",
                    ht=str(ht['Reactor'][2:]),
                    condition=ht['Process conditions/Comments'].strip(),
                    sample=sample['Sample'],
                    feed=ht['Carbon source'],
                    starting_media=ht['Carbon source'],
                )
            all_samples[f"{md['sample']}_HT{md['ht']}"] = md

    paths = dict()
    for dirpath, dirnames, filenames in os.walk(args.data_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            basename = os.path.basename(full_path)
            if basename in all_samples:
                paths[basename] = full_path


    counts = Counter()

    sample_set = set(all_samples.keys())
    for sample in all_samples:
        md = all_samples[sample]
        metadata = ",".join(f"{k}={v}" for k, v in md.items())
        if sample not in paths:
            continue
        time_sample = sample.split('_')[0]
        counts[time_sample] += 1
        indir = paths[sample]
        print(f"lmcontrol crop {args.arguments} -m \"{metadata}\" {indir} {os.path.join(args.outdir, time_sample, sample)}")
        sample_set.remove(sample)


    for sample in counts:
        print(f"{sample}\t{counts[sample]}", file=sys.stderr)

if __name__ == '__main__':
    main()
