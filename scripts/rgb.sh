CUDA_VISIBLE_DEVICES=0 ns-train  umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color last_sample \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 32768 \
 --pipeline.model.method rgb \
 --data data/processed/ajar  \
 --experiment-name "rgb" \
 --vis viewer+wandb --viewer.websocket-port 7007 \