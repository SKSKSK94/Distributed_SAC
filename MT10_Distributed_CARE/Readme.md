# Distributed CARE Implementation

### Introduction
1. This is pytorch implementation of `Distributed CARE(Contextual Attention-based REpresentation learning)` using `Metaworld MT10(Mutli-Task 10)` environment
2. You can choose modified version of CARE(called CARE(M)) by setting `use_modified_care` as `true` in `cfg/MT10_Distributed_CARE_cfg`. The modified contents of CARE(M) is two as below :
   1. Using weighted loss to balance the training of easy/hard tasks
   2. Change the position of mlp 

   (See the attached image)
<img src = "https://user-images.githubusercontent.com/73100569/130648481-cbaf5244-febf-4a4e-9d2d-466034cab7ed.png" width="800" height="600">
   
   You can choose original version of CARE(called CARE(O)) by setting `use_modified_care` as `false` in `cfg/MT10_Distributed_CARE_cfg`.

The reference paper is [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/abs/2102.06177)

### Default Loaded Model

Default loaded model is the trained `CARE(M)`.

### Results

1. Success Rate
<img src = "https://user-images.githubusercontent.com/73100569/130654451-6458293b-4be2-4f55-ae59-ae240a8566b7.png" width="500" height="300">
2. Video
<img src = "https://user-images.githubusercontent.com/73100569/130655417-e2795643-66a6-4eea-b37d-4f9ced7266ce.gif">