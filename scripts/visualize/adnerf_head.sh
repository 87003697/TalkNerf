# adnerf dataset
export CUDA_VISIBLE_DEVICES=4
python3.7 test.py \
    --config-name adnerf_head \
    resume=True \
    train=False \
    precache_rays=False \
    data.dataset.preload_image=False \
    resume_from=/home/mazhiyuan/code/talknerf/outputs/2022-05-13-adnerf/adnerf-head-040-att-0/checkpoints/epoch54_weights.pth \
    test=compare