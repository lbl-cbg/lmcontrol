bash scripts/byol_train_simple.sh 
bash scripts/byol_predict_simple.sh 
lmcontrol prep-viz /pscratch/sd/n/niranjan/output/byol_output/viz_pkg.npz /pscratch/sd/n/niranjan/output/byol_predictions/prediction.npz
lmcontrol emb-viz /pscratch/sd/n/niranjan/output/byol_output/viz_pkg.npz