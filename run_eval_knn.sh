CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_knn_c.py \
    --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
    --patch_size 8 \
    --pretrained_weights models/dino_deitsmall8_pretrain_full_checkpoint.pth \
    --dump_features output/knn_features/vit_s8/ | tee vit_s8_knn.log &&

# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 eval_knn_c.py \
#     --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
#     --patch_size 16 \
#     --pretrained_weights models/dino_deitsmall16_pretrain_full_checkpoint.pth \
#     --dump_features output/knn_features/vit_s16/ | tee vit_s16_knn.log 