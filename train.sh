#!/bin/bash
#SBATCH -p i64m1tga800u
#SBATCH -J componerf
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

#GPU节点计算池	i64m1tga800u	50台   Host:gpu1-[1-65]   CPU: Intel 2*8358P,32C, 2.6GHz   Memory:1024GB   GPU: 8*A800   System Disk: SSD 2*960GB   OS：Ubuntu   15台   Host:gpu2-[1-15]   CPU: Intel 2*8358P,32C, 2.6GHz   Memory:1024GB   GPU: 8*A800   System Disk: SSD 2*960GB   Data Disk：SSD 6*1.92 TB   OS：Ubuntu	限时7天，用户总共使用GPU 8卡，每个用户提交16个任务
# i64m1tga40u	14台   CPU: Intel 2*8358P,32C, 2.6GHz   Memory:1024GB   System Disk: SSD 2*960GB   GPU: 8*A40   OS：Ubuntu	限时7天，用户总共使用GPU 8卡，每个用户提交16个任务
# long_gpu	与i64m1tga800u队列的资源共用	限时14天，用户总共使用GPU 8卡，每个用户提交16个任务
# 应急队列资源	emergency_gpu	与i64m1tga800u队列的资源共用	计划划分5-20%（可按需调节）的GPU集群和GPU集群的资源，限时2小时，任务空闲时大家都可以使用，如有应急任务执行，现有的任务完毕后将不接收普通任务，执行应急任务（应急任务执行时间可按需调节）
# Debug测试	debug	2台   CPU: Intel 2*8358P,32C, 2.6GHz   Memory:1024GB   System Disk: SSD 2*960GB   GPU: 8*A40   OS：Ubuntu	Debug CPU和GPU资源：适用用户对CPU、CUDA 、软件的适配，代码的调试调优，针对特殊容器环境镜像调优及教学实训。

#模板介绍
#第1行为shell脚本固定格式
#第2行SBATCH -p 为指定分区为i64m512u（目前slurm按机器型号分为i64m512u,i64m512r,a128m512u,i64m1tga800u，i96m3tu几个分区，其中i表示Intelcpu,a表示amd cpu，64、128表示每台服务器的cpu核心，m512表示512g内存，具体机器型号参考硬件资源章节）
#第3行SBATCH -J为指定作业名称，自定义配置
#第4行SBATCH --ntasks-per-node为每个节点运行多少核心（跨节点计算时必须配置该参数）
#第5行-n 128指定总的核心数
#第6行指定标准输出文件
#第7行指定错误输入文件
module load anaconda3
module load cuda/11.3

nvidia-smi
source /hpc2ssd/softwares/anaconda3/bin/activate 
conda activate componerf

# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/abls_configs/compo_nerf/apple_and_banana_no_global.yaml
# python train.py --config_path demo_configs/compo_nerf_recomp/target/table_apple.yaml
# python train.py --config_path demo_configs/compo_nerf_recomp/target/table_apple.yaml
# python train.py --config_path demo_configs/compo_nerf_recomp/target/table_chess.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/table_wine.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/apple_and_banana.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/astronaut.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/bed_room.yaml

python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/chess.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/eiffel.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/football_and_basketball.yaml

# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/glass_balls.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/pisa_tower.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/pyramid.yaml

# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/teddy_monkey.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/tesla.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/vase_flower.yaml
# python train.py --config_path /hpc2hdd/home/hbai965/workdir/CompoNeRF/demo_configs/compo_nerf/whale.yaml

