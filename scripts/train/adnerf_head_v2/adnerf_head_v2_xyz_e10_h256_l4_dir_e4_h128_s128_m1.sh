export CUDA_VISIBLE_DEVICES=7
python train.py \
    --config-name nerfgan_bgmask \
    data.dataset.preload_image=False \
    renderer=adnerf_head_v2 \
    raysampler=drvnerf \
    implicit_function=adnerf_v2_xyz_e10_h256_l8_dir_e4_h128 \
    precache_rays=False \
    data.dataloader.batch_size=1 \
    implicit_function.n_layers_xyz=4 \
    implicit_function.render_size=[128,128] \
    losses.mask=1 \
    losses.l1_bg=0 \
    losses.l1_torso=0 \
    losses.l1_face=10 \
    visualization.visdom_env='adnerf_v2_xyz_e10_h256_l4_dir_e4_h128_s128_m1' \
    checkpoint_epoch_interval=1