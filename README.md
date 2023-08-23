# EvenNICER-SLAM

This is a repository based on NICE-SLAM.
Please basically follow the README in their repo (https://github.com/cvg/nice-slam).

## Installation

Create and activate an environment `evennicer-slam` by running:
```bash
conda env create -f environment.yaml
conda activate evennicer-slam
```

## Datasets

### Replica
Download the data as below and the data is saved into the `./Datasets/Replica` folder.
```bash
bash scripts/download_replica.sh
```
Then you need an extra GT event image dataset (generated using ESIM), which can be downloaded here: https://polybox.ethz.ch/index.php/s/JEUIwGWFjdaWK4x (password: evennicer)

The directory `./Datasets` should look like this:
```bash
Datasets
├── Replica
│   ├── office0
│   │   └── results
│   ├── office1
│   │   └── results
│   ├── office2
│   │   └── results
│   ├── office3
│   │   └── results
│   ├── office4
│   │   └── results
│   ├── room0
│   │   └── results
│   ├── room1
│   │   └── results
│   └── room2
│       └── results
└── replica_gt_png
    ├── office0
    ├── office1
    ├── office2
    ├── office3
    ├── office4
    ├── room0
    ├── room1
    └── room2
```
and you can run EvenNICER-SLAM:
```bash
python -W ignore run.py configs/Replica/room0.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec.ply`, where the unseen regions are culled using all frames.

If you have access to SLURM cluster, you can also run the bash script to make life easier:
```bash
mkdir -p ./output/log ./output/wandb
sbatch ./scripts/slurm_run.sh
```
The log file can be tailed using command:
```bash
tail -F output/log/<job_ID>.out
```
The experiments can be monitored using wandb.

### Real-world Dataset from RPG

This dataset is based on https://rpg.ifi.uzh.ch/direct_event_camera_tracking/index.html
The original dataset from RPG provides RGBE and GT poses, and a point cloud.
The depth maps are rendered using Open3D from the point clouds (and thus noisy).
The GT poses are also manually transformed to fit the required coordinate system for SLAM application, so they might be numerically noisy as well.

Download the dataset here and unzip under the directory `./Datasets`:
https://polybox.ethz.ch/index.php/s/MVQmhiVniF2UzEi

There are some recordings with rendered depth maps ready. (recording3, 4)
For recordings for which depth maps are not ready, depth maps need to be rendered manually (code to be updated...)

Run EvenNICER-SLAM on the RPG dataset by specifying the correct config file. For example:
```bash
python -W ignore run.py configs/rpg/recording4.yaml --output output
```

The bash script for SLURM is also updated. It can be run by typing:
```bash
mkdir -p ./output/log ./output/wandb
sbatch ./scripts/slurm_run.sh
```
