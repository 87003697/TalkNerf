export CUDA_VISIBLE_DEVICES=7
python train.py \
    --config-name nerfgan_bgmask \
    data.dataset.preload_image=False \
    renderer=mlp_toy \
    implicit_function=mlp_toy_e10_h256_l2 \
    precache_rays=False \
    implicit_function.n_layers_xyz=8 \
    losses.mask=0 \
    losses.l1_bg=0 \
    losses.l1_torso=0 \
    visualization.visdom_env='mlp_toy_e10_h256_l8_only_face' \
    checkpoint_epoch_interval=1