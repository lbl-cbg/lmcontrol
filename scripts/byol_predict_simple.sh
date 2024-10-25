INPUT_DIR=${1:?"Please provide the input data directory"};
OUTPUT_DIR=${2:?"Please provide the directory to save results to"};
CKPT=${3:?"Please provide the checkpoint to use for inference"};
lmcontrol infer-byol \
    -c $CKPT \
    --prediction \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT3/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT9/all_processed.npz \
    -o /pscratch/sd/n/niranjan/output/byol_predictions/prediction.npz \
    --mask \
    time 