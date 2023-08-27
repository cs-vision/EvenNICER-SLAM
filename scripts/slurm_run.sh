#!/bin/bash
#SBATCH  --output=/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/output/log/%j.out
#SBATCH  --error=/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/output/log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --constraint='geforce_gtx_titan_x'

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 
source /itet-stor/myamaguchi/net_scratch/conda/etc/profile.d/conda.sh
conda activate evennicer-slam

# Add RPG dataset in the future
datasets=("Replica")
replica_scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
output_affix="/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/output"

method="evennicer-slam"
dataset=${datasets[0]}
scene_name="office3"

# Edit this to distinguish different configs
run_suffix="50_150_Asynchronous"

# Run single or array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    python -W ignore run.py configs/${dataset}/${scene_name}.yaml --output ${output_affix}/${method}/${dataset}/${scene_name}-${run_suffix}
else
    scene_name=${replica_scenes[$SLURM_ARRAY_TASK_ID]} # start with 0
    python -W ignore run.py configs/${dataset}/${scene_name}.yaml --output ${output_affix}/${method}/${dataset}/${scene_name}-${run_suffix}
fi

# Send some noteworthy information to the output log
echo ""
echo "Job Comment:     test EvenNICER-SLAM"
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     ${JOB_START_TIME}"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0