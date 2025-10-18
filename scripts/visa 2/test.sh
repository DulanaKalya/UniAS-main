torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  train_val.py -e \
  --config ./configs/visa_part2_multi.yaml \
  --exp-path ./experiments/visa_part2_multi