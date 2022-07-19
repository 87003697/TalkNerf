export CUDA_VISIBLE_DEVICES=2
python train.py \
    --config-name nerfgan_bgmask \
    data.dataset.preload_image=False \
    precache_rays=False \
    visualization.visdom_env='render_bgmask_e10_h256_l2_nerfgan_lr' \
    checkpoint_epoch_interval=1