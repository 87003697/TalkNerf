# lego dataset
export CUDA_VISIBLE_DEVICES=5
python3.7 train.py \
    --config-name lego \
    precache_rays=False \
    # resume=True \
    # resume_from=/home/mazhiyuan/code/talknerf/outputs/2022-04-15/lego_run_3_part2/checkpoints/epoch6180_weights.pth
