CUDA_VISIBLE_DEVICES=3 ns-train umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color random \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 8192 \
 --pipeline.model.method rgb \
  --pipeline.model.implementation tcnn \
 --data data/processed/ajar  \
 --experiment-name "rgb" \
 --vis wandb --viewer.websocket-port 7007 \

# ns-render camera-path --load-config outputs/hotdog-t0.4-k6-specular/umhsnerf/2025-03-07_034249/config.yml --camera-path-filename /home/perezpnf/unsupervised-hyperspectral-nerf/data/processed/hotdog/camera_paths/2025-03-07-05-06-45.json --output-path renders/hotdog/2025-03-07-05-06-45.mp4