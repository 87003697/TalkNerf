export CUDA_VISIBLE_DEVICES=5
python test.py \
    --config-name headnerf \
    data.dataset.preload_image=False \
    implicit_function.render_size=[128,128] \
    precache_rays=False \
    resume=True \
    train=False \
    resume_from=
    test=compare