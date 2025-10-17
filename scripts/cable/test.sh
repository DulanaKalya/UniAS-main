torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  train_val.py -e \
  --config ./configs/mvtec_ad_cable.yaml \
  --exp-path ./experiments/mvtec_ad_cable
