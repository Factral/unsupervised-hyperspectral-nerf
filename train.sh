ns-train  umhsnerf --pipeline.model.far-plane 1000 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color last_sample --data data/ajar_adapted  \
 --vis viewer+wandb --viewer.websocket-port 7007 \
 #--load-dir outputs/output/umhsnerf/2024-11-07_111837/nerfstudio_models
