# lego dataset
export CUDA_VISIBLE_DEVICES=6
python3.7 test.py \
    --config-name lego \
    resume=True \
    train=False \
    precache_rays=False \
    resume_from=/home/mazhiyuan/code/talknerf/outputs/2022-04-24/lego_run_3_part3/checkpoints/epoch17910_weights.pth \
    test.mode=export_video \
    test=circular
