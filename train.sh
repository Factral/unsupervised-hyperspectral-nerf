ns-train  umhsnerf \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color last_sample \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.model.method spectral \
 --data data/ajar_adapted2  \
 --experiment-name "ajar_spectral_gradient_onlyspectral" \
 --vis viewer+wandb --viewer.websocket-port 7007 \