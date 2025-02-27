CUDA_VISIBLE_DEVICES=3 ns-train umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.num_classes 7 \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color random \
 --pipeline.model.spectral_loss_weight 4.0 \
 --pipeline.model.temperature 0.2 \
 --pipeline.model.pred_dino False \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 8192 \
 --pipeline.model.method rgb+spectral \
 --data data/processed/ajar \
 --experiment-name "ajar" \
 --vis viewer+wandb --viewer.websocket-port 7007 \

 #3.8.18