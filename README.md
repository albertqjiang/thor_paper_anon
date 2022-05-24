# Anonymized code for the submitted paper

## Pre-training

Follow the instructions in the [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) codebase 
for pre-training a 700m model.

## Data and environment

Follow the instructions in the [Portal-to-Isabelle](https://github.com/albertqjiang/Portal-to-ISAbelle) codebase
for preparing the Isabelle environment the data for fine-tuning. 

For both pre-training and fine-tuning, make sure to change the transformer configuration files according the paper's description.

## Evaluation
Use solvers/bfs_solvers for evaluation. It implements the best first search algorithm we described in the paper.
It establishes a connection with the Isabelle server through grpc.