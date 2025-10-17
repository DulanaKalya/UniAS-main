torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  train_val.py \
  --config ./configs/visa_multi.yaml \
  --exp-path ./experiments/visa_multi