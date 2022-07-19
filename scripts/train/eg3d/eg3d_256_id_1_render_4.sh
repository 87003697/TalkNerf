export CUDA_VISIBLE_DEVICES=3
python train.py \
    --config-name eg3d \
    data.dataset.preload_image=False \
    precache_rays=False \
    renderer.tri_plane_sizes=[256] \
    visualization.visdom_env='eg3d_256_id_1_render_4_density_no_sig' \
    checkpoint_epoch_interval=10 \
    renderer=eg3d_id_1_render_4
