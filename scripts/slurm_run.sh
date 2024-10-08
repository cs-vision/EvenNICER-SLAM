#!/bin/bash
#SBATCH  --output=/srv/beegfs02/scratch/data_shichen/data/EvenNICER-SLAM/output/log/%j.out
#SBATCH  --error=/srv/beegfs02/scratch/data_shichen/data/EvenNICER-SLAM/output/log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --constraint='titan_xp'

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 
source /srv/beegfs02/scratch/data_shichen/data/conda/etc/profile.d/conda.sh
conda activate evennicer-slam

datasets=("Replica" "rpg")
replica_scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
rpg_scenes=("recording3" "recording4") # These two recordings should be available out of the box, others can be added if needed
output_affix="/srv/beegfs02/scratch/data_shichen/data/EvenNICER-SLAM/output"

method="evennicer-slam"
dataset=${datasets[1]}
scene_name="recording4"

# Edit this to distinguish different configs
run_suffix="recording4_test"

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