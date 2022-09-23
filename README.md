# Object Detection in an Urban Environment

## Project overview

This project is to develop object detection model to train a CNN (Convolutionnal Neural Network) using Waymo TFRecord. The model will detect objects and classify them, vehicle, pedestrien, and cyclists in an image. This development process includes data spliting for training and validation, train CNN model, evaluate results and adust parameters. In this project we use SSD Resnet 50 model as baseline, and try to improve this model using data argumentation and adjust optimizer.

This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

## Setup
This section should contain a brief description of the steps to follow to run the code for this repository.

## Dataset
For this project, we are using data from the [Waymo Open dataset](https://waymo.com/open/). The data we use for training, validation and testing is organized as follow:
```
/home/workspace/data/
    - train: contain the train data (86 files)
    - val: contain the val data (10 files)
    - test - contains 3 files to test our model and create inference videos
```

The `training` and `val` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling. Note that (/home/workspace) is a location of where the files was extracted using git clone.

#### Data split
Using Waymo datasets, we have tried to split data following 90:10 ratio. Actually, 86 files in `training` and 10 files in `val` folders.

### Dataset analysis

#### Exploratory Data Analysis
Before we start model training, we check ramdom images in the dataset.
We extract ramdom 10 images and result is [here](https://github.com/n-kamata/ObjectDetection/blob/master/Exploratory%20Data%20Analysis.ipynb).
We can see the figure using this nootebook:

```
open jupytor nootebook
- Exploratory Data Analysis.ipynb
```

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/eda_example.png" width="400">
Fig.1 An example od exploratory data analysis. 

the mean of the numper of vehcles in an image is 17.4, that of pedestrien is 1.4, and that of cyclists is 0.1. It is rarely to see cyclists in the datasets, but we don't want to ignore the cyclists, because this project is for SDC, and ths system should be safe.

### Cross validation

This section should detail the cross validation strategy and justify your approach.




### Training
We choose reference model. We have change config and train/eval. The results are saved in this file structure:

```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training result with the unchanged config file
    - ex1/ - Improved config and result (Added data augmentation)
    - ex2/ - Improved config and result (Changed optimiser)
    - ex3/ - Improved config and result (Changed optimizer parameters
    - label_map.pbtxt
    ...
```
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

For this project, we decided to use SSD Resnet 50 as a baseline to make it easy to
study how to improve model. 


Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

This result is not good. The loss is greater than 10 and we want to improve this. Fist, we tried data augmentation to improve learing efficiency and avoid overfitting.

### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

#### Apply data Augmentation
We apply data augmentation and train again. This result is [here](https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/result.txt).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/da_example.png" width="400">
Fig2. Data augmentation result.

Data augmentation is in Fig. 2. Compareing with Fig. 1, we can understand the difference.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/lr.png" width="400">

Even if we apply the data augmentation loss is still learge. Next, we tried change optimizer.

#### Optimizer and learningrate
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/lr.png" width="400">

Taking this result into account, optimizer algorithm gets better, because even if , RMS loss is better than Ada loss. Next we adjust parameter about learing rate with RMS.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/rl.png" width="400">

We restrict learing rate at begining og epoch, loss is restricted and it keeps around 1.0.

I do not have experiment using different architectuire from SSD, but we can try more using SSD.
- Increase the number of data
- Vary of scene data
- Vary of classes
- Optimization, cross validation
- others

#### Animation
This is a result of trained SSD.
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/animation.gif" width="400">