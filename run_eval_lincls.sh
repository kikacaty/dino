# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
#     --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
#     --patch_size 8 \
#     --pretrained_weights models/dino_deitsmall8_pretrain_full_checkpoint.pth \
#     --lincls_checkpoint models/lincls/dino_deitsmall8_linearweights.pth | tee vit_s8_lincls.log

# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
#     --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
#     --patch_size 16 \
#     --pretrained_weights models/dino_deitsmall16_pretrain_full_checkpoint.pth \
#     --lincls_checkpoint models/lincls/dino_deitsmall16_linearweights.pth | tee vit_s16_lincls.log

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
    --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
    --patch_size 16 --arch vit_base --avgpool_patchtokens True --n_last_blocks 1\
    --pretrained_weights models/dino_vitbase16_pretrain_full_checkpoint.pth \
    --lincls_checkpoint models/lincls/dino_vitbase16_linearweights.pth | tee vit_b16_lincls.log