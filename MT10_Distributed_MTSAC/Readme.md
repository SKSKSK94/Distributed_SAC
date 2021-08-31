# Distributed MT-SAC Implementation

### Introduction
1. This is pytorch implementation of `Distributed MT-SAC` using `Metaworld MT10(Mutli-Task 10)` environment
2. You can choose modified version of MT-SAC(`MT-SAC with Weighted Loss`) by setting `use_weighted_loss` as `true` in `cfg/MT10_Distributed_MTSAC_cfg`. The modified contents of `MT-SAC with Weighted Loss` is just one as below :
   - Using weighted loss to balance the training of easy/hard tasks
   
   You can choose original version of MT-SAC(called `MT-SAC`) by setting `use_weighted_loss` as `false` in `cfg/MT10_Distributed_MTSAC_cfg`.

The reference paper is [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://arxiv.org/abs/1910.10897)

### Default Loaded Model

Default loaded model is the trained `MT-SAC with Weighted Loss`.
