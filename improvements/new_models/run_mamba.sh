#!/bin/bash
# Mamba, chosen setting (validation): d=64, lr 1e-3, dropout 0.2, wd 0.1, flip+noise augmentation. One process per seed.
cd /home/user/tv
run() { [ -f e2/e2_mamba_s$1.json ] || TV_THREADS=2 TV_AUG=1 TV_DROP=0.2 TV_WD=0.1 TV_D=64 TV_LR=1e-3 python e2/e2_neural.py mamba --steps 1000 --seed_list $1 --out e2/e2_mamba_s$1.json > e2/mamba_s$1.log 2>&1; }
run 1 & run 2 & wait
run 3 & run 4 & wait
echo ALLDONE >> e2/mamba_done.txt
