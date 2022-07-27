export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config-name headnerf \
    data.dataset.preload_image=False \
    losses.mask=10 \
    losses.l1_bg=0 \
    losses.l1_torso=0 \
    losses.l1_face=10 \
    losses.gan=1 \
    losses.percept=0 \
    implicit_function.render_size=[32,32] \
    precache_rays=False \
    visualization.visdom_env='headnerf_xyz_e10_h256_l6_dir_e4_h128_mf32_32_512_l_f10_m10_g1_p0' \
    # test=null \
