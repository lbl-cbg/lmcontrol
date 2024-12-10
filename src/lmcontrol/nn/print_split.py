from itertools import chain, permutations
import sys
import os

base_dir = "/pscratch/sd/n/niranjan/tar_ball/segmented_square_96/"
ht_files = {
    "HT1": [
        os.path.join(base_dir, "S4/S4_HT1/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT1/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT1/all_processed.npz")
    ],
    "HT2": [
        os.path.join(base_dir, "S4/S4_HT2/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT2/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT2/all_processed.npz")
    ],
    "HT3": [
        os.path.join(base_dir, "S4/S4_HT3/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT3/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT3/all_processed.npz")
    ],
    "HT4": [
        os.path.join(base_dir, "S4/S4_HT4/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT4/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT4/all_processed.npz")
    ],
    "HT5": [
        os.path.join(base_dir, "S4/S4_HT5/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT5/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT5/all_processed.npz")
    ],
    "HT6": [
        os.path.join(base_dir, "S4/S4_HT6/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT6/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT6/all_processed.npz")
    ],
    "HT7": [
        os.path.join(base_dir, "S4/S4_HT7/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT7/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT7/all_processed.npz")
    ],
    "HT8": [
        os.path.join(base_dir, "S4/S4_HT8/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT8/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT8/all_processed.npz")
    ],
    "HT9": [
        os.path.join(base_dir, "S4/S4_HT9/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT9/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT9/all_processed.npz")
    ],
    "HT10": [
        os.path.join(base_dir, "S4/S4_HT10/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT10/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT10/all_processed.npz")
    ],
    "HT11": [
        os.path.join(base_dir, "S4/S4_HT11/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT11/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT11/all_processed.npz")
    ],
    "HT12": [
        os.path.join(base_dir, "S4/S4_HT12/all_processed.npz"),
        os.path.join(base_dir, "S10/S10_HT12/all_processed.npz"),
        os.path.join(base_dir, "S14/S14_HT12/all_processed.npz")
    ]
}


replicates = [
    [ht_files["HT1"], ht_files["HT2"], ht_files["HT3"]],
    [ht_files["HT4"], ht_files["HT5"], ht_files["HT6"]],
    [ht_files["HT7"], ht_files["HT8"], ht_files["HT9"]],
    [ht_files["HT10"], ht_files["HT11"], ht_files["HT12"]]
]

def print_split(split):
    combined = list(chain(*chain(split)))
    print(" ".join(combined))

perms = list(permutations([0, 1, 2]))

def base6(i):
    q, c1 = divmod(i, 6)
    q, c2 = divmod(q, 6)
    q, c3 = divmod(q, 6)
    q, c4 = divmod(q, 6)
    return [c1, c2, c3, c4]

split_type = sys.argv[1] if len(sys.argv) > 1 else 'train'

if len(sys.argv) < 3:
    i = 2  # Default value
else:
    i = int(sys.argv[2])

indices = base6(i)

perms = [perms[j] for j in base6(i)]
split = list()
for p, r in zip(perms, replicates):
    split.append([r[j] for j in p])
split = list(zip(*split))

dict_mapping = { 'train': split[0], 'validation': split[1], 'test': split[2]}

print_split(dict_mapping[sys.argv[1]])
