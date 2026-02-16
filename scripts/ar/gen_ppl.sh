#!/bin/bash
#SBATCH -J ar_sample_eval                          # Job name
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
#SBATCH --export=ALL                                # export submission env to job


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
        --ckpt_path) ckpt_path="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_nodes) num_nodes="$2"; shift ;;
        --length) length="$2"; shift ;;
        --model) model="$2"; shift ;;
        --dataset) dataset="$2"; shift ;;
        --samples_path) samples_path="$2"; shift ;;
        --p_nucleus) p_nucleus="$2"; shift ;;
        --num_samples) num_samples="$2"; shift ;;
        --hf_cache_dir) hf_cache_dir="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

seed=0
srun python -u -m main \
  mode=sample_eval \
  algo=ar \
  loader.eval_batch_size=$batch_size \
  model=${model} \
  data=${dataset} \
  +wandb.offline=true \
  model.length=${length} \
  sampling.num_samples=$num_samples \
  eval.compute_generative_perplexity=True \
  eval.checkpoint_path=${ckpt_path} \
  seed=$seed \
  trainer.num_nodes=${num_nodes} \
  huggingface.instantiate_eval_model=True \
  eval.generated_samples_path=${samples_path} \
  sampling.p_nucleus=${p_nucleus} \
  sampling.kv_cache=True \
  huggingface.cache_dir=${hf_cache_dir}
