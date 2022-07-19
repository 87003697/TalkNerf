export CUDA_VISIBLE_DEVICES=7
python train.py \
    --config-name adnerf_head \
    data.dataset.preload_image=False \
    precache_rays=False \
    raysampler.rect_sample_rate=0.95 \
    nosmo_epoches=0 \
    visualization.visdom_env='adnerf_095_att_0_x2_bg_v3' \
    optimizer=adam_lambda_adnerf_x2
    # test=null \
