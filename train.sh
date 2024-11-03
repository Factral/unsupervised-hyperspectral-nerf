ns-train  umhsnerf --pipeline.model.far-plane 1000 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color last_sample --data data/output --vis viewer+wandb --viewer.websocket-port 7007