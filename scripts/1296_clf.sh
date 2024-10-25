#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=2:30:00
#SBATCH --array=0  # Only one job in the array (since we're only using one name)
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_01_00.err
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_10_18_01_00.out
#SBATCH -J opta_10_18_01_00  # Job name

# Assign custom name (SLURM_ARRAY_TASK_ID is replaced with the custom name)
task_name="test_10_23_01_00"

# # # Run the script with the task_name
# lmcontrol sup-train \
#     --mode classification \
#     --training \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT1/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT4/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
#     --validation \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
#     --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     --epochs 5 \
#     --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ \
#     --stop_wandb \
#     -n 100 \
#     testing0 \
#     sample


# training
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT10/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT4/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT7/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT10/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT1/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT7/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT10/all_processed.npz \


# validation
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/all_processed.npz \

# Get the best checkpoint
best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/ | \
    grep 'checkpoint-' | \
    sort -t '-' -k3,3 -g | \
    tail -n 1)

best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${task_name}/$best_ckpt"

# Run the prediction step using the best checkpoint
lmcontrol sup-predict \
    --mode classification \
    --prediction \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT9/all_processed.npz \
    --checkpoint $best_ckpt_path \
    -o /pscratch/sd/n/niranjan/output/prediction_${task_name}.npz \
    -n 100 \
    sample \
    # -save_emb \
    # --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${task_name} \
    # --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/


# prediction
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT12/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT3/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT6/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT9/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT12/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT3/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT6/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT9/all_processed.npz \
#     /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT12/all_processed.npz \



# !/bin/bash
# SBATCH -A m4521
# SBATCH -C gpu
# SBATCH -q shared
# SBATCH -c 32
# SBATCH --gpus-per-task=1
# SBATCH -n 1
# SBATCH --time=1:30:00
# SBATCH --array=54
# SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_%a 
# SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/opta_1296_combis/optatune.%A_%a
# SBATCH -J opta_1296_combis

# SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}
# SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-10_04_01} #delete

# TRAIN=`python /pscratch/sd/n/niranjan/lmcontrol/src/lmcontrol/nn/print_split.py train ${SLURM_ARRAY_TASK_ID}`
# VALIDATION=`python /pscratch/sd/n/niranjan/lmcontrol/src/lmcontrol/nn/print_split.py validation ${SLURM_ARRAY_TASK_ID}`
# TEST=`python /pscratch/sd/n/niranjan/lmcontrol/src/lmcontrol/nn/print_split.py test ${SLURM_ARRAY_TASK_ID}`

# lmcontrol opta-clf-train \
#     --training $TRAIN \
#     --validation $VALIDATION \
#     --checkpoint /pscratch/sd/n/niranjan/output/optatune/opta_${SLURM_ARRAY_TASK_ID}/ \
#     --epochs 50 \
#     --outdir /pscratch/sd/n/niranjan/output/optatune/opta_${SLURM_ARRAY_TASK_ID}/ \
#     -n 9500 \
#     --early_stopping \
#     testing0 \
#     time

# best_ckpt=$(ls /pscratch/sd/n/niranjan/output/optatune/opta_${SLURM_ARRAY_TASK_ID}/ | \
#     grep 'checkpoint-' | \
#     sort -t '-' -k3,3 -g | \
#     tail -n 1)

# best_ckpt_path="/pscratch/sd/n/niranjan/output/optatune/opta_${SLURM_ARRAY_TASK_ID}/$best_ckpt"

# lmcontrol opta-clf-predict \
#     --prediction $TEST \
#     --checkpoint $best_ckpt_path \
#     -o /pscratch/sd/n/niranjan/output/prediction_${SLURM_ARRAY_TASK_ID}.npz \
#     -n 9500 \
#     time \
#     --save_misclassified /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/misclassified_${SLURM_ARRAY_TASK_ID} \
#     --save_confusion /pscratch/sd/n/niranjan/output/optatune/ambr01_misclassify/ \
    
# Add -save_emb below time in predict. --predict is troublemaker .Here time is expected to below -n in any case
