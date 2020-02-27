# META

Implementation of the paper "META-Learning State-based Eligibility Traces for More Sample-Efficient Policy Evaluation" [1].

The "ringworld" tests use our implemented version of the environment.

This repository also contains our reproduced $\lambda$-greedy algorithm [2], with some additional tools and MATLAB scripts to draw the figures showed in the paper [1].

## References
[1] [Zhao et al., META-Learning State-based Eligibility Traces for More Sample-Efficient Policy Evaluation, 2019](https://arxiv.org/abs/1904.11439)

[2] [White and White, A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning, 2016](https://arxiv.org/abs/1607.00446)

## Requirements

  * Python 3.6+
  * Numpy, Numba
  * OpenAI Gym
  * Dependent python modules

## Cite

Please kindly cite our work if necessary:

```
@inproceedings{zhao2020meta,
title={META-Learning State-based Eligibility Traces for More Sample-Efficient Policy Evaluation},
author={Zhao, Mingde and Luan, Sitao and Porada, Ian and Chang, Xiao-Wen and Precup, Doina},
booktitle = {International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS)},
year = {2020},
url={https://arxiv.org/abs/1904.11439}
}
```
