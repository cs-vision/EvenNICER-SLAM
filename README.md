# EvenNICER-SLAM

This is a repository based on NICE-SLAM.
Please basically follow the README in their repo (https://github.com/cvg/nice-slam).

## Datasets

### Replica
Download the data as below and the data is saved into the `./Datasets/Replica` folder.
```bash
bash scripts/download_replica.sh
```
Then you need an extra GT event image dataset (generated using ESIM), which can be downloaded here: https://polybox.ethz.ch/index.php/s/JEUIwGWFjdaWK4x
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
mkdir ./output/log
sbatch ./scripts/slurm_run.sh
```
The log file can be tailed using command:
```bash
tail -F output/log/<job_ID>.out
```
The experiments can be monitored using wandb.

### Real-world Dataset from RPG

Coming soon...

