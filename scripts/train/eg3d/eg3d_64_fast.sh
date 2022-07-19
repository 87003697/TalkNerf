export CUDA_VISIBLE_DEVICES=0
python train.py \
    --config-name eg3d \
    data.dataset.preload_image=False \
    precache_rays=False \
    renderer.tri_plane_sizes=[64] \
    visualization.visdom_env='eg3d_64_fast_density_no_sig' \
    checkpoint_epoch_interval=20 \
    optimizer.lr=0.01 \
    renderer.norm_momentum=0.01 \
    losses.gan=0 \
    losses.id=0 \
    losses.percept=0 \




    
