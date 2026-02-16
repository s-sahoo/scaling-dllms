#!/bin/bash

# Usage:
# ./auto_resubmit.sh -n 5 -d PREVIOUS_JOB_ID -D 6 -h 768 -s 1000000 -b 64 <file.sub>

# Grab command line options
# n: Number of times to submit the job
# d: Previous job ID
# D: model depth
# l: learning rate
N_CALLS=1
PREV_JOBID=""
model='6M'
flops=-1
batch_size=64
num_nodes=1
tag=""
length=2048
cluster='eos'
dataset='nvidia'
phase='1'
while getopts "n:d:m:b:N:t:f:c:D:p:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
    d) PREV_JOBID=$OPTARG;;
    m) model=$OPTARG;;
    b) batch_size=$OPTARG;;
    N) num_nodes=$OPTARG;;
    t) tag=$OPTARG;;
    f) flops=$OPTARG;;
    c) cluster=$OPTARG;;
    D) dataset=$OPTARG;;
    p) phase=$OPTARG;;
  esac
done


if [ $cluster == 'eos' ]; then
    CKPT_BASE=/lustre/fsw/convai_convaird_nemo-speech/users/ssahoo
    if [ $dataset == 'slim_pajama' ]; then
        DATA_DIR=/lustre/fsw/llmservice_nemo_reasoning/data/slim_pajama_tokenized_new_3
    else
        DATA_DIR=/lustre/fsw/llmservice_nemo_reasoning/data/nemotron_phase${phase}_1t/
    fi
    extra_args=""
else
    CKPT_BASE=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_difflm/users/susahoo
    if [ $dataset == 'slim_pajama' ]; then
        DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_difflm/data/slim_pajama_tokenized_new_3
    else
        DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_difflm/data/nemotron_phase${phase}_1t
    fi
    extra_args="--gres=gpu:8"
fi

if [ $flops == -1 ]; then
    wandb_name=${tag}-${model}-phase${phase}
    ckpt_dir=${CKPT_BASE}/${tag}/${model}-phase${phase}
else
    wandb_name=${tag}-${model}-${flops}
    ckpt_dir=${CKPT_BASE}/${tag}/${model}-${flops}
fi


# Grab the .sub file to run
SUBFILE=${@:$OPTIND:1}
if [[ -z $SUBFILE ]]; then
    echo "ERROR: sub file not provided"
    echo "Usage: $(basename "$0") [flags] [sub file]"
    exit 1
fi

if [ ! -f ${SUBFILE} ]
then
    echo "ERROR: ${SUBFILE} not found!"
    echo "Usage: $(basename "$0") [flags] [sub file]"
    exit 1
fi

echo "Submit dependent jobs"
echo "  script   : ${SUBFILE}"
echo "  num jobs : ${N_CALLS}"
echo "  model    : ${model}"
echo "  flops    : ${flops}"
echo "  ckpt_dir : ${ckpt_dir}"
echo "  wandb_name : ${wandb_name}"
echo "  num_nodes : ${num_nodes}"
# Repeat calls

# Check if a job with name $wandb_name is running or pending for user susahoo
RUNNING_JOBS=$(squeue -u susahoo -h -o "%j %T" | awk -v t="$wandb_name" '$1 == t && ($2 == "RUNNING" || $2 == "PENDING")')
if [[ "$SUBFILE" == *train* && "$PREV_JOBID" == "" ]]; then
    if [[ ! -z "$RUNNING_JOBS" ]]; then
        echo "A job with name '$wandb_name' is already running or pending for user susahoo:"
        echo "$RUNNING_JOBS"
        exit 0
    fi
fi


for (( i = 1; i <= $N_CALLS; i++ ))
do
    if [ -z $PREV_JOBID ]
    then
        echo "Submitting job ${i}"
        OUTPUT=$(sbatch -J ${wandb_name} \
            $extra_args \
            -N $num_nodes $SUBFILE \
            --ckpt_dir $ckpt_dir \
            --wandb_name $wandb_name \
            --model $model \
            --flops $flops \
            --batch_size $batch_size \
            --num_nodes $num_nodes \
            --length $length \
            --data_dir $DATA_DIR \
            --dataset $dataset \
            )
    else
        echo "Submitting job ${i} w/ dependency on jobid ${PREV_JOBID}"
        OUTPUT=$(sbatch --dependency=afterany:${PREV_JOBID} -J ${wandb_name} \
            $extra_args \
            -N $num_nodes $SUBFILE \
            --ckpt_dir $ckpt_dir \
            --wandb_name $wandb_name \
            --model $model \
            --flops $flops \
            --length $length \
            --batch_size $batch_size \
            --num_nodes $num_nodes \
            --data_dir $DATA_DIR \
            --dataset $dataset \
            )
    fi
    PREV_JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
done

squeue --start -u $(whoami) -l
