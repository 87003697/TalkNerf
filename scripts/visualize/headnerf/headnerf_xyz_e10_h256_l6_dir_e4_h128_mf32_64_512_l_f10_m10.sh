export CUDA_VISIBLE_DEVICES=2
python test.py \
    --config-name headnerf \
    data.dataset.preload_image=False \
    implicit_function.render_size=[64,64] \
    precache_rays=False \
    visualization.visdom_env='headnerf_xyz_e10_h256_l6_dir_e4_h128_mf32_64_512_l_f10_m10' \
    resume=True \
    train=False \
    resume_from=    
    test=compare