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

### Hints generation
You can find hints generation procedures for each algorithm in `generate_data.py`.