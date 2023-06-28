#  Lightweight Multi-Branch Network for Person Re-Identification

[![Paper](https://img.shields.io/badge/arXiv-2101.10774-important)](https://arxiv.org/abs/2101.10774)
[![IEEE](https://img.shields.io/badge/IEEE-ICIP2021-blue)](https://ieeexplore.ieee.org/abstract/document/9506733)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lightweight-multi-branch-network-for-person/person-re-identification-on-cuhk03-detected)](https://paperswithcode.com/sota/person-re-identification-on-cuhk03-detected?p=lightweight-multi-branch-network-for-person)


**News**: We support *Vehicle Re-Identification* on VeRi and AICity now. See below for instructions and pretrained models.

![](/utils/LightMB.png)

This repo supports easy dataset preparation, including Market-1501, DukeMTMC-ReID, CUHK03, MOT17, sota deep neural networks and various options(tricks) for reid, easy combination of different kinds of loss function, end-to-end training and evaluation and less package requirements.

List of functions
- Warm up cosine annealing learning rate
- Random erasing augmentation
- Cutout augmentation
- Drop Block and Batch Erasing
- Label smoothing(Cross Entropy loss)
- Triplet loss
- Multi-Simulatity loss
- Focal loss
- Center loss
- Ranked list loss
- Group Loss
- Different optimizers
- Attention modules
- BNNeck

Inplemented networks:
- Lightweight Multi-Branch Network(LightMBN), which we proposed
- PCB [[link]](https://arxiv.org/pdf/1711.09349.pdf)
- MGN [[link]](https://arxiv.org/abs/1804.01438)
- Bag of tricks [[link]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
- OSNet [[link]](https://arxiv.org/abs/1905.00953)
- Batch Drop Block(BDB) for Person ReID [[link]](https://arxiv.org/abs/1811.07130)

## Getting Started
The designed code architecture is concise and easy explicable, where the file engine.py defines the train/ test process and main.py controls the overall epochs, and the folders model, loss, optimizer including respective parts of neural network.

The user-friendly command-line module argparse helps us indicate different datasets, networks, loss functions, and tricks as we need, the detailed options/configurations are described in the bottom of this page.

If you don't have any dataset yet, run 
```
git clone https://github.com/jixunbo/ReIDataset.git
```
to download Market-1501, DukeMTMC, CUHK03 and MOT17.

Create a virtual environment and install torch and torchvision. Then, install the remaining requirements:

```
pip install -r requirements.txt
```

To implement our Lightweight Multi-Branch Network with Multi-Similarity loss, run

```
python [path to repo]/main.py --datadir [path to datasets] --data_train market1501 --data_test market1501 --model LMBN_n --batchid 6 --batchimage 8 --batchtest 32 --test_every 20 --epochs 130 --loss 0.5*CrossEntropy+0.5*MSLoss --margin 0.7 --nGPU 1 --lr 6e-4 --optimizer ADAM --random_erasing --feats 512 --save '' --if_labelsmooth --w_cosine_annealing
```

You can also use the config file:

````
python [path to repo]/main.py --config [path to repo]/lmbn_config.yaml --save ''
````

All logs, results and parameters will be saved in folder 'experiment'.

`--data_train` and `--data_test` specify the name of train/test dataset, respectively.

`--batchid 8` and `--batchimage 6` indicate that each batch contrains 8 targets, and each target has 6 different images in the batch. The resulting total batch size is `batchid * batchimage = 48`.

`--epochs` is the epochs we'd like to train, while `--test_every 10` means evaluation will be excuted in every 10 epochs, the parameters of network and optimizer are updated after every every evaluation. 

For the LightMBN model, we used two kinds of backbonse, ResNet50 (`LightMBN_R`), and OSNet (`LightMBN_N`). OSNet has fewer parameters but achieves higher performance.

For more options, click [here](OPTIONS.md).

### Results
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lightweight-multi-branch-network-for-person/person-re-identification-on-cuhk03-labeled)](https://paperswithcode.com/sota/person-re-identification-on-cuhk03-labeled?p=lightweight-multi-branch-network-for-person)
|            Model | Market1501  | DukeMTMC-reID | CUHK03-D    | CUHK03-L    | VeRi |
| ---------------: | ----------- | ------------- | ----------- | ----------- | ---- |
|         LightMBN | 96.3 (91.5) | 92.1 (83.7)   | 85.4 (82.6) | 87.2 (85.1) | ---  |
|        + re-rank | 96.8 (95.3) | 93.5 (90.2)   | 90.2 (90.6) | 91.3 (92.2) | ---  |
| LightMBN(ResNet) | 96.1 (90.5) | 90.5 (82.0)   | 81.0 (79.2) | 83.7 (82.5) | ---  |
|              BoT | 94.2 (85.4) | 86.7 (75.8)   |             |             | ---  |
|              PCB | 95.1 (86.3) | 87.6 (76.6)   |             |             | ---  |
|              MGN | 94.7 (87.5) | 88.7 (79.4)   |             |             | ___  |

Note, Rank-1(mAP), the results are produced by our repo **without re-ranking**, models and configurations may differ from original paper.

Additionally, the evaluation metric method is the same as bag of tricks [repo](https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/reid_metric.py).

For training on cuhk03 dataset with MS Loss, the batch size of 8 x 8 would be recommended.

### Pre-trained models
and correpondent config files can be found [here](https://1drv.ms/u/s!Ap1wlV4d0agrao4DxXe8loc_k30?e=I9PJXP) .

If you have pretrained model and config file, run
```
python [path to repo]/main.py --test_only --config [path to repo]/lmbn_config.yaml --pre_train [path to pretrained model]
```
to see the performance of the model.

If you would like to re-implement Bag of Tricks, run
```
python [path to repo]/main.py --datadir [path to datasets] --data_train market1501 --data_test market1501 --model ResNet50 --batchid 16 --batchimage 4 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_40_70 --loss 0.5*CrossEntropy+0.5*Triplet --margin 0.3 --nGPU 1 --lr 3.5e-4 --optimizer ADAM --random_erasing --warmup 'linear' --if_labelsmooth
```
or 
```
python [path to repo]/main.py --config [path to repo]/bag_of_tricks_config.yaml --save ''
```

If you would like to re-inplement PCB with powerful training tricks, run
```
python [path to repo]/main.py --datadir [path to datasets] --data_train Market1501 --data_test Market1501 --model PCB --batchid 8 --batchimage 8 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_50_80_110 --loss 0.5*CrossEntropy+0.5*MSLoss --margin 0.7 --nGPU 1 --lr 5e-3 --optimizer SGD --random_erasing --warmup 'linear' --if_labelsmooth --bnneck --parts 3
```

Note that, the option '--parts' is used to set the number of stripes to be devided, original paper set 6.

And also, for MGN model run
```
python [path to repo]/main.py --datadir [path to datasets] --data_train Market1501 --data_test Market1501 --model MGN --batchid 16 --batchimage 4 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_50_80_110 --loss 0.5*CrossEntropy+0.5*Triplet --margin 1.2 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --warmup 'linear' --if_labelsmooth
```

### Resume Training

If you want to resume training process, we assume you have the checkpoint file 'model-latest.pth', run
```
python [path to repo]/main.py --config [path to repo]/lmbn_config.yaml --load [path to checkpoint]
```
Of course, you can also set options individually using argparse command-line without config file.

## Easy Implementation  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14aRebdOqJSfNlwXiI5USOQBgweckUwLS)
Our code can be implemented easily online without installing any package or requirement even GPU in your own computer thanks to Google Colab, all the packages we need are Colab standard pre-installed packages.

Open this [notebook](https://colab.research.google.com/drive/14aRebdOqJSfNlwXiI5USOQBgweckUwLS), following the steps there and you can see the training process and results.

Please be sure that your are using Google's powerful GPU(Tesla P100 or T4).

The whole training process(120 epochs) takes ~9 hours.
If you are hard-core player ^ ^ and you'd like to try different models or options, see Get Started as above.

## Vehicle Re-Identification

LightMBN performs well on vehicle re-identification datasets with some hyperparameter adjustments. The VeRi results were obtained by running

```
python main.py --datadir [...] --data_train veri --data_test veri --model LMBN_n --batchid 8 --batchimage 8 --batchtest 32 --test_every 40 --epochs 130 --loss "0.5*CrossEntropy+0.5*MSLoss" --margin 0.7 --nGPU 1 --lr 6e-4 --optimizer ADAM --random_erasing --feats 512 --save '' --if_labelsmooth --w_cosine_annealing --height 222 --width 222 --transforms "random_flip+random_crop+brightness_adjustment"
```

## Acknowledgments
The codes was built on the top of  [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) and [MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch) , we thank the authors for sharing their code publicly.

## Reference

If you want to cite LightMBN, please use the following BibTex:

```
@INPROCEEDINGS{9506733,
  author={Herzog, Fabian and Ji, Xunbo and Teepe, Torben and Hörmann, Stefan and Gilg, Johannes and Rigoll, Gerhard},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Lightweight Multi-Branch Network For Person Re-Identification}, 
  year={2021},
  volume={},
  number={},
  pages={1129-1133},
  doi={10.1109/ICIP42928.2021.9506733}}
```

