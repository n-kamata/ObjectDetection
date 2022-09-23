# Object Detection in an Urban Environment

## Project overview

This project is to develop object detection model to train a CNN (Convolutionnal Neural Network) using Waymo TFRecord. The model will detect objects and classify them, vehicle, pedestrien, and cyclists in an image. This development process includes data spliting for training and validation, train CNN model, evaluate results and adust parameters. In this project we use SSD Resnet 50 model as baseline, and try to improve this model using data argumentation and adjust optimizer.
Object detection is an important component of self driving car systems, because even if other modules, localization, planner, and contoller works, an incident can be caused by miss detection.

## Setup

### Data
For this project, we are using data from the [Waymo Open dataset](https://waymo.com/open/). 

### Data split

The data we use for training, validation and testing is organized as follow:
```
/home/workspace/data/
    - train: contain the train data (86 files)
    - val: contain the val data (10 files)
    - test - contains 3 files to test our model and create inference videos
```

The `training` and `val` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling. Note that (/home/workspace) is a location of where the files was extracted using git clone.

Using Waymo datasets, we have tried to split data following 90:10 ratio. Actually, 86 files in `training` and 10 files in `val` folders.

### Visualization
We use jupyter nootebook to visualize our analisys and test result with with Firefox or Google chrome.

```
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root

```
### Model selection and training
For the training process we use Tensorflow object detection API. The default configuration is called `pipeline.config` and it contains information about training data, parameters, data augumentation, and so on. 

First, we want to choose API. Next, we download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

Once we choose API, we create config files and modify it to improve the model.

```
python edit_config.py --train_dir /home/workspace/data/waymo/train/ --eval_dir /home/workspace/data/waymo/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```

We create `pipeline.config`, we move the config to exeriment folder and rename `pipeline_new.config`.

```
mv pipeline_new.config /home/workspace/experiments/reference/
```

We train this model using this command.

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
To evaluate the model, the following command is used.
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```
While testing, we can measure intermediate result using Tensorboard using Firefox or Google crhome.
```
python -m tensorboard.main --logdir experiments/
```
When we try other models or other `pipeline.config`, we create folders in `/home/workspace/experiment`. We saved data in the following structure. When we finished experiment, we save data like  `ex1`, `ex2`, ... When we execute commands fot train and evaluation, please replace `reference` with `ex*` folder.

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

## Dataset

### Dataset analysis
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

Cross validation is a method to separate dataset for train and evaluation, and measure model performance. This project didn't use this validation, but it is good idea to use `KFold` in `sklearn.model_selection` to execute cross validation. The following is 

## Training

We choose reference model. We have change config and train/eval. The results are saved in this file structure:

#### Reference experiment
For this project, we decided to use SSD Resnet 50 as a baseline to make it easy to
study how to improve model. 

This result of baseline is not good. The loss is greater than 10 and we want to improve this. Fist, we tried data augmentation to improve learing efficiency and avoid overfitting. Next, we adjested leaning rate annealing.

### Improve on the reference
This project strategy is like this.
- Data aurmentation
- Learning rate annealing

We choose loss as metrics, and set target total loss 1.0 or less.

#### Apply data Augmentation
We apply data augmentation and train again. This result is [here](https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/result.txt).

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/da_example.png" width="400">
Fig2. Data augmentation result.

Data augmentation is in Fig. 2. Compareing with Fig. 1, we can understand the difference.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/lr.png" width="400">

Even if we apply the data augmentation loss is still learge. 

```
INFO:tensorflow:    + Loss/total_loss: 6.968129
```

Next, we tried change optimizer.

#### Optimizer and learning rate
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/lr.png" width="400">

```
INFO:tensorflow:    + Loss/total_loss: 1.638250
```

Since learning rate is same and RMS loss gets better than Ada loss, RMS should be used than Ada optimizer.

Next we adjust parameter about learing rate with RMS.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/rl.png" width="400">

```
INFO:tensorflow:    + Loss/total_loss: 1.118462
```

We restrict learing rate at begining og epoch, loss is restricted and it keeps around 1.0. we almost meet our goal.

I do not have experiment using different architectuire other than SSD, but we can try more using SSD.
- Increase the number of data
- Vary of scene data
- Vary of classes
- Optimization, cross validation
- others

#### Animation

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/animation.gif" width="400">
This is a result of trained SSD. Object detecter developed by this project can work in night.