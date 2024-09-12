#!/bin/bash

base_dir="/mnt/nfs_asia"
input_file="${base_dir}/csvs/negation_dataset/combined/cc12m_images_extracted_pos_neg.csv"
output_base="${base_dir}/csvs/negation_dataset/cc12m_images_pos_neg_filtered"

# Total rows to process
TOTAL_ROWS=10003877
# Number of jobs (one per GPU)
NUM_JOBS=8
# Number of rows per job (divide by all jobs, including the last one)
ROWS_PER_JOB=$((TOTAL_ROWS / NUM_JOBS))

# Function to calculate start and end index for a job
get_indices () {
    local job_index=$1
    local start_index=$((job_index * ROWS_PER_JOB))
    
    # For the last job, set index_end to -1 (process remaining rows)
    if [ "$job_index" -eq $((NUM_JOBS - 1)) ]; then
        local end_index=-1
    else
        local end_index=$(((job_index + 1) * ROWS_PER_JOB))
    fi
    
    echo "$start_index $end_index"
}

source /mnt/nfs_asia/miniconda3/etc/profile.d/conda.sh
conda activate vLLM
export TMPDIR=/mnt/nfs_asia/tmp
export PIP_CACHE_DIR=/mnt/nfs_asia/pip_cache
export HF_HOME="/mnt/nfs_asia/tmp/huggingface"
export HF_TOKEN="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"

# Loop over each job (one for each GPU)
for i in $(seq 0 $((NUM_JOBS - 1))); do
    # Get start and end indices for the job
    read START_INDEX END_INDEX <<< $(get_indices $i)
    
    # Output file with start and end indices in the filename
    output_file="${output_base}_${START_INDEX}_${END_INDEX}.csv"
    
    # Run the job on the corresponding GPU
    echo "Running job $i on GPU $i with rows $START_INDEX to $END_INDEX"
    
    CUDA_VISIBLE_DEVICES=$i python filter_negative_objects.py --input_file $input_file --output_file $output_file --index_start $START_INDEX --index_end $END_INDEX > logs/filter_${i}.log 2>&1 &
done

echo "All jobs launched."
