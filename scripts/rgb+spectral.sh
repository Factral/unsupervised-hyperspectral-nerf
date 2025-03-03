CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 ns-train umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.num_classes 7 \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color black \
 --pipeline.model.spectral_loss_weight 5.0 \
 --pipeline.model.temperature 0.4 \
 --pipeline.model.pred_dino False \
 --pipeline.model.pred_specular True \
 --pipeline.model.load_vca True \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 8192 \
 --pipeline.model.method rgb+spectral \
 --data  data/hsnerf/surface_optics/origami/masked_processed_dataset \
 --experiment-name "origami-t0.4-k2" \
 --vis viewer+wandb --viewer.websocket-port 7007 \

 #3.8.18