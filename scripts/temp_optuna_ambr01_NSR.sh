#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH --time=1:30:00
#SBATCH --array=0-10%4
#SBATCH -e /pscratch/sd/n/niranjan/error/clf_error_rep/optuna_multilabels/optatune.%A_%a 
#SBATCH -o /pscratch/sd/n/niranjan/error/clf_error_rep/optuna_multilabels/optatune.%A_%a
#SBATCH -J optuna_multilabels

#python -c "import optuna; optuna.delete_study(study_name='study', storage='sqlite:////pscratch/sd/n/niranjan/output/optatune/multilabels_trials_11_12_01_00/study.db')"

task_name="11_12_01_02_${SLURM_ARRAY_TASK_ID}"

echo "Running task with ID: ${SLURM_ARRAY_TASK_ID}"
# task_name="test_multilabels_11_12_01_00"

lmcontrol tune \
    -t 1 \
    --training \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT10/all_processed.npz \
    --validation \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/all_processed.npz \
    -e 25 \
    -n 9500 \
    /pscratch/sd/n/niranjan/output/optatune/multilabels_trials_${task_name} \
    time \
    feed \
