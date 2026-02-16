# [Scaling Beyond Masked Diffusion Language Models]()
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Jean-Marie Lamercier](https://scholar.google.com/citations?user=dJFuXCQAAAAJ&hl=fr), [Justin Deschenaux](https://jdeschena.com), [Zhihan Yang](https://zhihanyang2022.github.io), [Jingyu Liu](https://jingyu6.github.io),
[John Thickstun](https://johnthickstun.com), [Ante Jukic](https://scholar.google.com/citations?user=ZleK6ccAAAAJ&hl=en)

[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](http://s-sahoo.github.io/duo)
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2506.10892v1)


# 1.7B Models

## Checkpoints will be released on March 1st, 2026.

## Training

```
./auto_resubmit.sh -n 10  -m 2121M -b 2 -N 16 -D nvidia -p 2 -t ar scripts/ar/train.sh 
./auto_resubmit.sh -n 10  -m 2121M -b 2 -N 16 -D nvidia -p 2 -t mdlm scripts/mdlm/train.sh 
```

## Scaling Laws

```
./auto_resubmit.sh -n 5 -m 6M -f 6 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 34M -f 6 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 66M -f 6 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 85M -f 6 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 113M -f 6 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 170M -f 6 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 336M -f 6 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 666M -f 6 -b 8 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 


./auto_resubmit.sh -n 5 -m 19M -f 10 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 48M -f 10 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 85M -f 10 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 170M -f 10 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 206M -f 10 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 666M -f 10 -b 8 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 944M -f 10 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 



./auto_resubmit.sh -n 5 -m 66M -f 30 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 142M -f 30 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 231M -f 30 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 472M -f 30 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 666M -f 30 -b 8 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 944M -f 30 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 1233M -f 30 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 1476M -f 30 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 2121M -f 30 -b 2 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 



./auto_resubmit.sh -n 5 -m 142M -f 60 -b 32 -N 1 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 231M -f 60 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 472M -f 60 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 666M -f 60 -b 8 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 1233M -f 60 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 1476M -f 60 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 2121M -f 60 -b 2 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 


./auto_resubmit.sh -n 5 -m 231M -f 100 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 472M -f 100 -b 16 -N 2 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 666M -f 100 -b 8 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 1233M -f 100 -b 4 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
./auto_resubmit.sh -n 5 -m 2121M -f 100 -b 2 -N 4 -t chinchilla-mdlm-v1 scripts/mdlm/train_slim_mdlm.sh 
```