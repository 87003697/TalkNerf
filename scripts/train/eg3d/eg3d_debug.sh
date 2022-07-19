export CUDA_VISIBLE_DEVICES=1
python train.py \
    --config-name eg3d \
    data.dataset.preload_image=False \
    precache_rays=False \
    renderer.tri_plane_sizes=[64] \
    visualization.visdom_env='eg3d_debug' \
