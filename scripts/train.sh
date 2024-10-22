#!/bin/bash
#SBATCH -A m3513
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 10:00:00
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -e /pscratch/sd/n/niranjan/error/eo_report.%j #Note that doing this will save the erros and output to the same file ,and .j is the job number.
#SBATCH -o /pscratch/sd/n/niranjan/error/eo_report.%j
#SBATCH -J Run1

lmcontrol train-clf  \
    --training \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT10/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT10/all_processed.npz \
     --validation \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/all_processed.npz \
    -c /pscratch/sd/n/niranjan/output/checkpoint/model.ckpt \
    -e 10 \
    -o /pscratch/sd/n/niranjan/output/ \
    -n 20000 \
    testing0 \
    time



    # training
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT4/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT7/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT10/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT1/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT7/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT10/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT1/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT4/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT10/all_processed.npz \

    #validation
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/all_processed.npz \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/all_processed.npz \

#Note: to continue the same command, just add '\' at the end of line else if the command is new, do not add '/'.
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT11/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT5/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT8/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT11/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT2/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT8/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT11/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT2/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT5/ \
    # /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/ \