export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config-name nerfgan_bgmask \
    data.dataset.preload_image=False \
    precache_rays=False \
    implicit_function.n_layers_xyz=7 \
    visualization.visdom_env='render_bgmask_e10_h256_l7_nerfgan_lr' \
    checkpoint_epoch_interval=1