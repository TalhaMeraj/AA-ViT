#!/bin/bash
#SBATCH --job-name=resvit_train         # Job name
#SBATCH --output=resvit_%j.out          # Standard output and error log
#SBATCH --partition=defq                # Correct partition
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for single-GPU)
#SBATCH --gres=gpu:1                    # Request 1 GPU


# Optional: log some environment info
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Load conda environment
source ~/setup-conda
conda activate e_resvit


# Run the Python training script
python test.py \
    --dataroot /HPC/fnu.talha/2D_RGB_Full_processed/T1_T2_FLAIR_T1ce \
    --name T1_T2_Flair_T1ce_nc_3_FullBrats2021_ReTrain_L1tune_ErrorMap_RDEB_Gradient_EMAFFTLoss_batch_24_ResVit \
    --gpu_ids  0 \
    --model resvit_many \
    --which_model_netG resvit \
    --dataset_mode aligned \
    --norm batch \
    --which_direction AtoB \
    --output_nc 1 \
    --input_nc 3 \
    --fineSize 256 \
    --loadSize 256 \
    --checkpoints_dir checkpoints/ \
    --which_epoch latest \
    --how_many 23000 \
    --serial_batches \
    --results_dir results/