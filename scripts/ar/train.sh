#!/bin/bash
#SBATCH -J ar_2k                                # Job name
#SBATCH -A llmservice_nemo_difflm                  # llmservice_nemo_difflm, llmservice_nemo_reasoning, llmservice_nemo_speechlm, convai_convaird_nemo-speech
#SBATCH -p batch                                   # batch 
#SBATCH -o watch_folder/%x_%j.out                  # output file (%j expands to jobID)
#SBATCH --get-user-env                             # retrieve the users login environment
#SBATCH --open-mode=append                         # Do not overwrite logs
#SBATCH -N 2                                       # number of nodes # -N 16
#SBATCH -t 04:00:00                                # wall time  (04:00:00 for batch, 08:00:00 for backfill)
#SBATCH --exclusive                                # exclusive node access
#SBATCH --mem=0                                    # all mem avail
#SBATCH --mail-type=FAIL                           # only send email on failure
#SBATCH --ntasks-per-node=8                        # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit                               # Needed for pytorch


# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

nvidia-smi

# Set PyTorch CUDA allocator configuration to avoid memory fragmentation  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Additional memory management settings
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=1


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wandb_name) wandb_name="$2"; shift ;;
        --ckpt_dir) ckpt_dir="$2"; shift ;;
        --flops) flops="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_nodes) num_nodes="$2"; shift ;;
        --length) length="$2"; shift ;;
        --model) model="$2"; shift ;;
        --data_dir) data_dir="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ $flops == -1 ]; then
    # Configs corresponding to large scale model training 
    # args="trainer.max_steps=1280000 \
    #     loader.global_batch_size=1024 \
    #     optim.lr=4e-4 \
    #     optim.min_lr=1e-4 \
    #     trainer.limit_val_batches=100 \
    #     trainer.val_check_interval=50000 \
    #     lr_scheduler.warmup_steps=2000"
    args="trainer.max_steps=350000 \
        loader.global_batch_size=1024 \
        optim.lr=3e-4 \
        optim.min_lr=4e-5 \
        trainer.limit_val_batches=100 \
        trainer.val_check_interval=50000 \
        lr_scheduler.warmup_steps=2000 \
        data.mode=phase2"
else
    # Configs corresponding to scaling laws training 
    args="training.flops_1e18=${flops} \
        algo.adaLN=True \
        trainer.limit_val_batches=1000"
fi

# finetune_path=/lustre/fsw/convai_convaird_nemo-speech/users/ssahoo/ar/2121M-phase1/checkpoints/0-500000.ckpt
srun python -u -m main \
  algo=ar \
  loader.batch_size=$batch_size \
  loader.eval_batch_size=$(($batch_size / 2)) \
  model=${model} \
  ${args} \
  data=${dataset} \
  wandb.name=${wandb_name} \
  model.length=${length} \
  model.dropout=0.0 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  trainer.log_every_n_steps=1000 \
  data.cache_dir=${data_dir} \
  hydra.run.dir=${ckpt_dir} \
  trainer.num_nodes=${num_nodes} \
  loader.n_chunks=1 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000
