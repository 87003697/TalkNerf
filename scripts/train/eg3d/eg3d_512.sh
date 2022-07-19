export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config-name eg3d \
    data.dataset.preload_image=False \
    precache_rays=False \
    renderer.tri_plane_sizes=[512] \
    visualization.visdom_env='eg3d_512' \
    checkpoint_epoch_interval=10



    
