# [Scaling Beyond Masked Diffusion Language Models]()
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Jean-Marie Lamercier](https://scholar.google.com/citations?user=dJFuXCQAAAAJ&hl=fr), [Justin Deschenaux](https://jdeschena.com), [Zhihan Yang](https://zhihanyang2022.github.io), [Jingyu Liu](https://jingyu6.github.io),
[John Thickstun](https://johnthickstun.com), [Ante Jukic](https://scholar.google.com/citations?user=ZleK6ccAAAAJ&hl=en)

[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2506.10892v1)

# Update: 1.7B Checkpoints will be released on March 1st, 2026.

<p align="center">
  <img src="https://github.com/s-sahoo/scaling-dllms/blob/gh-pages/static/images/scaling.png" alt="graphical_abstract_updated_2" width="70%">
</p>


In this repo, we release the `state-of-the-art` diffusion language models:
1. **Masked Diffusion Model: MDLM**
    > [Sahoo et al., "Simple and Effective Masked Diffusion Language Model", NeurIPS 2024.](https://arxiv.org/abs/2406.07524)
2. **Uniform-state Diffusion Model: Duo**
    > [Sahoo et al., "The Diffusion Duality", ICML 2025.](https://arxiv.org/abs/2506.10892)
3. **AR-MDLM interpolating method: Eso-LMs**
    > [Sahoo et al., "Esoteric Language Models", ArXiv 2025.](https://arxiv.org/abs/2506.01928)

# Scaling Laws

For scaling-law experiments, set:
* Algorithm: `DIR = ar / duo / esolm / mdlm`
* Model size: `MODEL = 6M / 19M / ... / 2121M`  ([Full list](configs/flops))
* Training flops (`x1e18`):
`FLOPS = 6 / 10 / 30 / 60 / 100`
in the following command:
```
./auto_resubmit.sh -n 5 -m <MODEL> -f <FLOPS> -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/<DIR>/train_slim_mdlm.sh 
```

# 1.7B Models

## Training
`DIR = {ar/duo/esolm/mdlm}`
```
./auto_resubmit.sh -n 10  -m 2121M -b 2 -N 16 -D nvidia -p 2 -t ar scripts/<DIR>/train.sh 
```

## Evaluation

 1.7B Checkpoints will be released on March 1st, 2026.