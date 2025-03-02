CUDA_VISIBLE_DEVICES=1 ns-train umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.num_classes 2 \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color black \
 --pipeline.model.spectral_loss_weight 5.0 \
 --pipeline.model.temperature 0.2 \
 --pipeline.model.pred_dino False \
 --pipeline.model.pred_specular False \
 --pipeline.model.load_vca True \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 8192 \
 --pipeline.model.method rgb+spectral \
 --data data/hsnerf/basil_white \
 --experiment-name "basil-t0.4-k2" \
 --vis viewer+wandb --viewer.websocket-port 7007 \

 #3.8.18