export CUDA_VISIBLE_DEVICES=1
python test.py \
    --config-name headnerf \
    data.dataset.preload_image=False \
    implicit_function.render_size=[64,64] \
    precache_rays=False \
    resume=True \
    train=False \
    resume_from=/home/15288906612/codes/talknerf/outputs/2022-07-20/05-13-03/checkpoints/epoch73_weights.pth \
    test=compare