lmcontrol stratify-clf \
    -e 3\
    --S10 \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT1/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT2/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S10/S10_HT3/all_processed.npz \
    --S14 \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT6/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT4/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S14/S14_HT5/all_processed.npz \
    --S4 \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT7/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT8/all_processed.npz \
    /pscratch/sd/n/niranjan/tar_ball/segmented_square_96/S4/S4_HT9/all_processed.npz \
    --tr 0.6 \
    --val 0.2 \
    --te 0.2 \
    -c /pscratch/sd/n/niranjan/output/stratify_output/checkpoint/model_simple.ckpt \
    -o /pscratch/sd/n/niranjan/output/stratify_output \
    -oe /pscratch/sd/n/niranjan/output/stratify_output/embeddings \
    -n 200 \
    time
