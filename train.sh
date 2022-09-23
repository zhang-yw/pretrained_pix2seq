#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path /nobackup/yiwei/coco/images --pix2seq_lr --large_scale_jitter --rand_target $@ --batch_size 16 --epochs 50
