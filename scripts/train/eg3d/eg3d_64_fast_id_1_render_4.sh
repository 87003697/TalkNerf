export CUDA_VISIBLE_DEVICES=4
python train.py \
    --config-name eg3d \
    data.dataset.preload_image=False \
    precache_rays=False \
    renderer.tri_plane_sizes=[64] \
    visualization.visdom_env='eg3d_64_fast_id_1_render_4_density_no_sig' \
    checkpoint_epoch_interval=20 \
    optimizer.lr=0.01 \
    renderer.norm_momentum=0.01 \
    losses.gan=0 \
    losses.id=0 \
    losses.percept=0 \
    renderer=eg3d_id_1_render_4




    