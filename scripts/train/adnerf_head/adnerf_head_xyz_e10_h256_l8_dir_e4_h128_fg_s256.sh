export CUDA_VISIBLE_DEVICES=6
python train.py \
    --config-name adnerf_head \
    data.dataset.preload_image=False \
    precache_rays=False \
    raysampler=adnerf_v2 \
    raysampler.rect_sample_rate=0.95 \
    nosmo_epoches=0 \
    implicit_function=adnerf_xyz_e10_h256_l8_dir_e4_h128_s64 \
    implicit_function.render_size=[256,256] \
    visualization.visdom_env='adnerf_095_att_0_x2_e10_h256_l8_dir_e4_h128_fg_s256' \
    optimizer=adam_lambda_adnerf_x2
    # test=null \
