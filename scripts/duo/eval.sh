#!/bin/bash
#SBATCH -J duo_2k                                # Job name
#SBATCH -A llmservice_nemo_difflm                  # llmservice_nemo_reasoning, llmservice_nemo_speechlm, convai_convaird_nemo-speech
#SBATCH -p batch                                   # batch 
#SBATCH -o watch_folder/%x_%j.out                  # output file (%j expands to jobID)
#SBATCH --get-user-env                             # retrieve the users login environment
#SBATCH --open-mode=append                         # Do not overwrite logs
#SBATCH -N 1                                       # number of nodes # -N 16
#SBATCH -t 04:00:00                                # wall time  (04:00:00 for batch, 08:00:00 for backfill)
#SBATCH --exclusive                                # exclusive node access
#SBATCH --mem=0                                    # all mem avail
#SBATCH --mail-type=FAIL                           # only send email on failure
#SBATCH --ntasks-per-node=8                        # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit                               # Needed for pytorch


nvidia-smi

# Set PyTorch CUDA allocator configuration to avoid memory fragmentation  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Additional memory management settings
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=1


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wandb_name) wandb_name="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
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


srun python -u -m main \
  mode=ppl_eval \
  algo=duo_base \
  loader.batch_size=$batch_size \
  loader.eval_batch_size=$(($batch_size / 2)) \
  model=${model} \
  data=${dataset} \
  wandb.name=${wandb_name}-eval \
  model.length=${length} \
  eval.generate_samples=True \
  eval.checkpoint_path=${ckpt_path} \
  eval.compute_generative_perplexity=False \
  trainer.val_check_interval=10000 \
  trainer.limit_val_batches=1000 \
  trainer.log_every_n_steps=1000 \
  data.cache_dir=${data_dir} \
  trainer.num_nodes=${num_nodes} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000
