python -m torch.distributed.launch --nproc_per_node=1 eval_linear.py \
    --data_path /home/scratch.sysarch_nvresearch/chaowei/datasets/ILSVRC2012 \
    --pretrained_weights models/lincls/dino_deitsmall8_linearweights.pth