# CS224W Project: Graph Models for Predicting Future User Preferences in Music Genres

This repository contains the code for our CS224W project. You can find the associated Medium story here: ADD LINK.

In this project, we aim to improve temporal user–item interaction prediction, a common problem in recommender systems that are widely used by companies today, for example in the music industry. These systems must adapt to users' evolving listening patterns, taking into account both long-term preferences and recent interactions. We focus on predicting future music genre preferences of users based on their temporal listening behavior.

The `tgbn-genre` dataset, provided by the **Temporal Graph Benchmark** project (TGB),  captures these temporal dynamics through a bipartite temporal graph, where users and music genres form nodes, and timestamped edges encode "user listens at a certain time to this genre" interactions. Each edge carries a weight that denotes the proportion of the listened song belonging to that genre. The core task is dynamic node property prediction: which genres a user will interact with in the following week. 

## References

The code in this repository is adapted from the following sources:

- https://github.com/shenyangHuang/TGB
- https://github.com/yule-BUAA/DyGLib 
- https://github.com/allegro/allRank

For more details on how they were implemented, please refer directly to these repositories.

## Organization

This repository is structured into the following folders:

- `Baselines`: Code for the Moving Average (MA) and GCN baseline models
- `DyGFormer`: Implementation of the DyGFormer model with MA features
- `TGN`: Code for all models derived from or built on top of TGN
- `Shallow embedding initialization`: TO ADD

## Running the code

All notebooks in this repository are designed to run on Google Colab, which we used for all experiments. Training temporal graph models on the full graph is computationally expensive: TGN with MA features takes approximately 13 hours for 50 epochs on a T4 GPU in Colab.

For an overview of the project and a discussion of the models, please refer to the accompanying Medium story.
