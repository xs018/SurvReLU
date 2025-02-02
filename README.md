
![paper](figs/cikm2024.jpg)
# [CIKM '24] SurvReLU: Inherently Interpretable Survival Analysis via Deep ReLU Networks 
[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2407.14463)

## News :fire:
- **July 16, 2024:** Congratulations! This paper has been accepted by CIKM 2024  $\color{lightgreen} \textbf{[acceptance rate = 27\\%]}\%$ !

  
![Inter](figs/tree.png)
<img align="right" width="60%" height="100%" src="figs/performance.jpg">

> **Abstract.** 
Survival analysis models time-to-event distributions with censorship. Recently, deep survival models using neural networks have dominated due to their representational power and state-of-the-art performance. However, their "black-box" nature hinders interpretability, which is crucial in real-world applications. In contrast, "white-box" tree-based survival models offer better interpretability but struggle to converge to global optima due to greedy expansion. In this paper, we bridge the gap between previous deep survival models and traditional tree-based survival models through deep rectified linear unit (ReLU) networks. We show that a deliberately constructed deep ReLU network (SurvReLU) can harness the interpretability of tree-based structures with the representational power of deep survival models. 
Empirical studies on both simulated and real survival benchmark datasets show the effectiveness of the proposed SurvReLU in terms of performance and interoperability.

> **Network Architecture**  
![Method](figs/network.png)


# Usage
## 1. Prepare dataset
## 2. Dependencies
## 3. Train From Scratch

## Acknowledgements

This code is built on the top of [pycox](https://github.com/havakv/pycox) and [locally constant network](https://github.com/guanghelee/iclr20-locally-constant-nets), we thank their efficient and neat codebase. 

## Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@misc{sun2024survreluinherentlyinterpretablesurvival,
      title={SurvReLU: Inherently Interpretable Survival Analysis via Deep ReLU Networks}, 
      author={Xiaotong Sun and Peijie Qiu and Shengfan Zhang},
      year={2024},
      eprint={2407.14463},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14463}, 
}
```
