CUDA_VISIBLE_DEVICES=1 ns-train  umhsnerf \
 --machine.seed 42 \
 --log-gradients True \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color last_sample \
 --pipeline.datamanager.images-on-gpu True \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 8192 \
 --pipeline.model.method spectral \
 --pipeline.model.implementation tcnn \
 --data data/processed/ajar  \
 --experiment-name "spectral only try" \
 --vis viewer --viewer.websocket-port 7008 \