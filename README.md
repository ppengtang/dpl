# Deep Patch Learning for Weakly Supervised Object Classification and Discovery

By [Peng Tang](https://ppengtang.github.io/), [Xinggang Wang](http://mclab.eic.hust.edu.cn/~xwang/index.htm), Zilong Huang, [Xiang Bai](http://mclab.eic.hust.edu.cn/~xbai/), and [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu).

### Introduction

**Deep Patch Learning (DPL)** is a fast framework for object classification and discovery with deep ConvNets. 
 - It achieves state-of-the-art performance on object classification (Pascal VOC 2007 and 2012), and very competitive results on object discovery.
 - Our code is written by C++ and Python, based on [Caffe](http://caffe.berkeleyvision.org/) and [fast r-cnn](https://github.com/rbgirshick/fast-rcnn).

The paper has been accepted by Pattern Recognition. For more details, please refer to our [paper](http://dx.doi.org/10.1016/j.patcog.2017.05.001).

If you are focusing on weakly supervised object detection (or object discovery), you can also see our recent CVPR2017 work [OICR](https://github.com/ppengtang/oicr).

### Architecture

<p align="left">
<img src="images/architecture.jpg" alt="DPL architecture" width="900px">
</p>

### Results

|  | VOC2007 test *mAP* (classification) | VOC2007 trainval *CorLoc* (discovery) | VOC2012 test *mAP* (classification) | VOC2012 trainval *CorLoc* (discovery)
|:-------|:-----:|:-------:|:-------:|:-------:|
| DPL-AlexNet | 85.3 | 43.5 | 84.4 | 48.7 |
| DPL-VGG16 | 92.7 | 45.4 | 92.5 | 51.0 |

### Visualizations

<p align="left">
<img src="images/patterns.jpg" alt="Some pattern visualization results" width="800px">
</p>
Some pattern visualization results.

<p align="left">
<img src="images/visualizations.jpg" alt="Some detection visualization results" width="800px">
</p>
Some detection visualization results.

### License

DPL is released under the MIT License (refer to the LICENSE file for details).

### Citing DPL

If you find DPL useful in your research, please consider citing:

    @inproceedings{tang2017deep,
        author = {Tang, Peng and Wang, Xinggang and Huang, Zilong and Bai, Xiang and Liu, Wenyu},
        title = {Deep Patch Learning for Weakly Supervised Object Classification and Discovery},
        journal = {Pattern Recognition},
        volume = {},
        pages = {},
        year = {2017}
    }
    
### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation)
4. [Installation for training and testing](#installation-for-training-and-testing)
5. [Extra Downloads (selective search)](#download-pre-computed-selective-search-object-proposals)
6. [Extra Downloads (ImageNet models)](#download-pre-trained-imagenet-models)
7. [Usage](#usage)
8. [Trained models](#our-trained-models)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. MATLAB

### Requirements: hardware

1. NVIDIA GTX TITANX (~12G of memory)

### Installation (sufficient for the demo)

1. Clone the DPL repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/ppengtang/dpl.git
  ```

2. Build the Cython modules
    ```Shell
    cd $DPL_ROOT/lib
    make
    ```
    
3. Build Caffe and pycaffe
    ```Shell
    cd $DPL_ROOT/caffe-dpl
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j 8 && make pycaffe
    ```

### Installation for training and testing
1. Download the training, validation, test data and VOCdevkit

    ```Shell
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
    ```
2. Extract all of these tars into one directory named `VOCdevkit`

    ```Shell
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_18-May-2011.tar
    ```
3. It should have this basic structure

    ```Shell
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

4. Create symlinks for the PASCAL VOC dataset

    ```Shell
    cd $DPL_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

5. [Optional] follow similar steps to get PASCAL VOC 2012.

6. You should put the generated proposal data under the folder $DPL_ROOT/data/selective_search_data, with the name "voc_2007_trainval.mat", "voc_2007_test.mat", just as the form of [fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

7. The pre-trained models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). You should put it under the folder $DPL_ROOT/data/imagenet_models, just as the form of [fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

### Download pre-computed Selective Search object proposals

Pre-computed selective search boxes can also be downloaded for VOC2007 and VOC2012.

```Shell
cd $DPL_ROOT
./data/scripts/fetch_selective_search_data.sh
```

This will populate the `$DPL_ROOT/data` folder with `selective_selective_data`.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded.

```Shell
cd $DPL_ROOT
./data/scripts/fetch_imagenet_models.sh
```
These models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

### Usage

**Train** a DPL network. For example, train a VGG16 network on VOC 2007 trainval:

```Shell
./tools/train_net.py --gpu 1 --solver models/VGG16/solver.prototxt \
	--weights data/imagenet_models/$VGG16_model_name --iters 40000
```

**Test** a DPL network. For example, test the VGG 16 network on VOC 2007 test:

#### Classification
```Shell
./tools/test_net.py --gpu 1 --def models/VGG16/test_cls.prototxt \
	--net output/default/voc_2007_trainval/vgg16_dpl_iter_40000.caffemodel
```

#### Discovery
```Shell
./tools/test_net.py --gpu 1 --def models/VGG16/test_det.prototxt \
  --net output/default/voc_2007_trainval/vgg16_dpl_iter_40000.caffemodel \
  --imdb voc_2007_trainval --task det
```

Test output is written underneath `$DPL_ROOT/output`.

#### Evalution

To get results, put the results under the folder `$VOCdevkit/results/VOC2007/Main`.

For classification, run the matlab code eval_classification.m

For discovery, run the matlab code eval_discovery.m

### Our trained models

The models trained on PASCAL VOC 2007 can be downloaded from [here](https://drive.google.com/open?id=0B3md9Lbjsj_qVE8tRUNvcUhSaW8).

And on PASCAL VOC 2012 can be downloaded from [here](https://drive.google.com/open?id=0B3md9Lbjsj_qdGoyZFllRFN5a1k).
