# adnerf dataset
export CUDA_VISIBLE_DEVICES=4
python3.7 test.py \
    --config-name adnerf_head \
    resume=True \
    train=False \
    precache_rays=False \
    data.dataset.preload_image=False \
    resume_from=/home/mazhiyuan/code/talknerf/outputs/2022-05-13-adnerf/adnerf-head-070-att-0/checkpoints/epoch54_weights.pth \
    test.mode=evaluation 
    # +data.test_file=/home/mazhiyuan/code/talknerf/datasets/Obama/transforms_val.json \
    # +data.aud_file=/home/mazhiyuan/code/talknerf/datasets/Obama/aud.npy