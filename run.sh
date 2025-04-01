set -e
set -u

export CUDA_VISIBLE_DEVICES=3,4
gpunum=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

dataset=maps # maps just for test
dataroot=./datasets/$dataset
model_name=${dataset}_pix2pixgan
checkpoints_dir=./temp_checkpoints
model=pix2pix
netG=unet_256

# ddp + fp16
accelerate launch --config-file accelerate_ddp_fp16.yaml train.py --dataroot $dataroot --name $model_name --model $model --checkpoints_dir $checkpoints_dir \
    --direction BtoA --netG $netG --lambda_L1 100 --norm batch --pool_size 0 \
    --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
    --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
    --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop

# ddp + fp16
accelerate launch --config-file accelerate.yaml train.py --dataroot $dataroot --name $model_name --model $model --checkpoints_dir $checkpoints_dir \
    --direction BtoA --netG $netG --lambda_L1 100 --norm batch --pool_size 0 \
    --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
    --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
    --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop

# no ddp, no fp16
# python train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop

# python train.py --dataroot $dataroot --name $model_name --model pix2pix --checkpoints_dir $checkpoints_dir \
#     --direction BtoA --netG unet_256 --lambda_L1 100 --norm batch --pool_size 0 \
#     --input_nc 3 --batch_size 16 --preprocess none --num_threads 8 --n_epochs 150 \
#     --n_epochs_decay 0 --load_size 256 --crop_size 256 --lr_policy cosine \
#     --display_id -1 --no_html --lr 0.0002 --preprocess resize_and_crop --use_fp16
