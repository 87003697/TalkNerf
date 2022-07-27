export CUDA_VISIBLE_DEVICES=7
python train.py \
    --config-name headnerf \
    data.dataset.preload_image=False \
    losses.mask=0 \
    losses.l1_bg=0 \
    losses.l1_torso=0 \
    losses.l1_face=100 \
    losses.gan=0 \
    losses.percept=0.01 \
    implicit_function.render_size=[64,64] \
    precache_rays=False \
    visualization.visdom_env='headnerf_xyz_e10_h256_l6_dir_e4_h128_mf32_64_512_l_f100_m0_g0_p001' \
    # test=null \
