INPUT_DIR=${1:?"Please provide the input data directory"};
OUTPUT_DIR=${2:?"Please provide the directory to save results to"};
lmcontrol train-byol \
    --training \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
    --validation \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
    -o /pscratch/sd/n/niranjan/output/byol_output \
    -e 3 \
    byol_testing0 \
    time
