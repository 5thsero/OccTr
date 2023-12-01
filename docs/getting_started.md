# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train OccTr with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/occtr/occtr-cross.py 8
```

Eval OccTr with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/configs/occtr/occtr-cross.py ./path/to/ckpts.pth 8
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)