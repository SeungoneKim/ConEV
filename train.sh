python -m torch.distributed.launch --nproc_per_node=7 \
train_dense_encoder.py \
train_datasets=defeasible_snli_weakener_train \
dev_datasets=defeasible_snli_weakener_dev \
train=biencoder_default \
output_dir="/home/intern2/seungone/ConEV/checkpoints"
