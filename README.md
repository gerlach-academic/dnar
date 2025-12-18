# Discrete Neural Algorithmic Reasoning
This repository contains the code to reproduce the experiments from "Discrete Neural Algorithmic Reasoning" paper. 

## Setup
Before running the source code, make sure to install the project dependencies:
```bash
pip install -r requirements.txt
```

## Main experiments

### Algorithms
- Breadth-first search
- Depth-first search
- Minimum spanning tree (Prim's algorithm)
- Maximum Independent Set (randomized)
- Shortest paths (Dijkstra's algorithm)


### Train a single-task model
```bash
python train.py --config_path
python eval.py
```

And you may pass multiple seeds which will be trained sequentially:
```bash
python train.py --config_path --num_seeds N
```

For large graphics cards we allow running multiple seeds in parallel:
```bash
python train.py --config_path --num_seeds N --parallel --num_workers M
```

Furthermore you may the flag `multitask` so that a you may specify multiple configs or a multitask config:
```bash
python train.py --config_path config1,config2 --num_seeds N --multitask
```
or
```bash
python train.py --config_path multitask_config --num_seeds N --multitask
```

For options on how the config affects the runs, please see the base configuration file `base_config.py` and the specific configuration files in the `configs` folder.

### Hints generation
You can find hints generation procedures for each algorithm in `generate_data.py`.