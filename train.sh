#!/bin/sh		
#BSUB -J nerf
#BSUB -n 4     
#BSUB -q gpu         
#BSUB -gpgpu 1
#BSUB -o out.%J      
#BSUB -e err.%J  
#BSUB -W 48:00

# nvidia-smi
# source activate
# conda activate componerf
# module load cuda-11.4



#CUDA_VISIBLE_DEVICES=4 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/apple_and_banana.yaml
#CUDA_VISIBLE_DEVICES=5 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/bed_room.yaml
#CUDA_VISIBLE_DEVICES=6 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/chess.yaml
#CUDA_VISIBLE_DEVICES=7 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/glass_balls.yaml
#CUDA_VISIBLE_DEVICES=4 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/astronaut.yaml
#CUDA_VISIBLE_DEVICES=0 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/football_and_basketball.yaml
#CUDA_VISIBLE_DEVICES=6 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/pisa_tower.yaml
# CUDA_VISIBLE_DEVICES=7 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/pyramid.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/tabel_wine.yaml
# CUDA_VISIBLE_DEVICES=5 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/teddy_monkey.yaml
# CUDA_VISIBLE_DEVICES=6 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/whale.yaml
# CUDA_VISIBLE_DEVICES=7 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/tesla.yaml
# CUDA_VISIBLE_DEVICES=4 python scripts/train_compo_nerf.py --config_path demo_configs/compo_nerf/vase_flower.yaml

# CUDA_VISIBLE_DEVICES=6 python train.py --config_path abls_configs/compo_nerf/table_wine.yaml
CUDA_VISIBLE_DEVICES=7 python train.py --config_path abls_configs/compo_nerf/table_wine_box_learn.yaml
CUDA_VISIBLE_DEVICES=6 python train.py --config_path abls_configs/compo_nerf/table_wine_box_learn2.yaml

CUDA_VISIBLE_DEVICES=7 python train.py --config_path demo_configs/compo_nerf/whale.yaml
CUDA_VISIBLE_DEVICES=6 python train.py --config_path demo_configs/compo_nerf/computer_mouse.yaml
CUDA_VISIBLE_DEVICES=5 python train.py --config_path demo_configs/compo_nerf/teddy_monkey.yaml
CUDA_VISIBLE_DEVICES=4 python train.py --config_path demo_configs/compo_nerf/bed_room.yaml

