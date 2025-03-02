CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 ns-train instant-ngp-bounded \
  --machine.seed 42 \
  --log-gradients True \
  --pipeline.model.far-plane 1000 \
  --pipeline.model.near_plane 0.05 \
  --pipeline.model.background-color random \
  --pipeline.datamanager.images-on-gpu True \
  --pipeline.datamanager.patch-size 1 \
  --pipeline.datamanager.train-num-rays-per-batch 8192 \
  --max-num-iterations 100000 \
  --data data/processed/cbox_sphere \
  --vis viewer+wandb \
  --viewer.websocket-port 7007 \

#/home/perezpnf/miniconda3/envs/umhs/lib/python3.8/site-packages/nerfstudio/