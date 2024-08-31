# Feature-pairs Relevant Attention Metric Network for Few-shot Image 
Classification

This repository contains the reference Pytorch source code for the following paper:

Feature-pairs Relevant Attention Metric Network for Few-shot Image 
Classification



## Code environment
This code requires Pytorch 1.7.0 and torchvision 0.8.0 or higher with cuda support. It has been tested on Ubuntu 16.04. 

You can create a conda environment with the correct dependencies using the following command lines:
```
conda env create -f environment.yml
conda activate FRN
```

## Dataset
The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN).

CUB_200_2011 [Download Link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

## Train

* To train FRAMN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/FRAMN/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/FRAMN/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/FRAMN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/FRAMN/ResNet-12
    python ./test.py
```

## References

Thanks to  [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch) and  [Yassine](https://github.com/yassouali/SCL), for the preliminary implementations.

