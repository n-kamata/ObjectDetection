# Object Detection in an Urban Environment

## Project overview

This project is to develop object detection model to train a CNN (Convolutionnal Neural Network) using Waymo TFRecord. The model will detect objects and classify them, vehicle, pedestrien, and cyclists in an image. This development process includes data spliting for training and validation, train CNN model, evaluate results and adust parameters. In this project I use SSD Resnet 50 model as baseline, and try to improve this model using data argumentation and adjust optimizer.
Object detection is an important component of self driving car systems, because even if other modules, localization, planner, and contoller works, an incident can be caused by miss detection.

## Setup

### Data
For this project, I are using data from the [Waymo Open dataset](https://waymo.com/open/). 

### Data split

The data I use for training, validation and testing is organized as follow:
```
/home/workspace/data/
    - train: contain the train data (86 files)
    - val: contain the val data (10 files)
    - test - contains 3 files to test our model and create inference videos
```

The `training` and `val` folder contains file that have been downsampled: I have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling. Note that (/home/workspace) is a location of where the files was extracted using git clone.

Using Waymo datasets, I have tried to split data following 90:10 ratio. Actually, 86 files in `training` and 10 files in `val` folders.

### Visualization
I use jupyter notebook to visualize our analysis and test result with with Firefox or Google chrome.

```
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
```

### Development environment (Dependancy)
The follwing is a list of the environment (dependencies) used to execute this project. I want to install them using `pip install` command. This project is based on the [Udacity Github repository](https://github.com/udacity/nd013-c1-vision-starter), I can use this as reference too.
``` 
jupyter                       1.0.0
jupyter-client                5.3.4
jupyter-console               6.1.0
jupyter-core                  4.6.2
matplotlib                    3.4.1
tensorboard                   2.4.1
tensorboard-plugin-wit        1.8.0
tensorflow                    2.4.1
tensorflow-addons             0.12.1
tensorflow-datasets           4.2.0
tensorflow-estimator          2.4.0
tensorflow-gpu                2.3.1
tensorflow-hub                0.11.0
tensorflow-metadata           0.28.0
tensorflow-model-optimization 0.5.0
```

### Model selection and training
For the training process I use Tensorflow object detection API. The default configuration is called `pipeline.config` and it contains information about training data, parameters, data augmentation, and so on. 

First, I chose API. Next, I downloaded the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

Once I chose model, I create config files and modify it to improve the model.

```
python edit_config.py --train_dir /home/workspace/data/waymo/train/ --eval_dir /home/workspace/data/waymo/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```

I created `pipeline.config`, I moved the config to experiment folder and rename `pipeline_new.config`.

```
mv pipeline_new.config /home/workspace/experiments/reference/
```

I trained this model using this command.

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
To evaluate the model, the following command is used.
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```
While testing, I measured intermediate result using Tensorboard using Firefox or Google chrome.
```
python -m tensorboard.main --logdir experiments/
```
When I tried other models or other `pipeline.config`, I created folders in `/home/workspace/experiment`. I saved data in the following structure. When I finished experiment, I saved data like  `ex1`, `ex2`, ... When I execute commands fot train and evaluation, please replace `reference` with `ex*` folder.

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
Before I started model training, I checked random images in the dataset.
I extracted random 10 images and result was [here](https://github.com/n-kamata/ObjectDetection/blob/master/Exploratory%20Data%20Analysis.ipynb).
The figure can be checked using this notebook:

```
open jupytor notebook
- Exploratory Data Analysis.ipynb
```

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/eda_example.png" width="400">
This figure is an example od exploratory data analysis. Bounding boxes are ground truth. The datasets has 3 classes, vehicle (colored red), pedestrian (green), and cyclist (blue).

The mean of the number of vehicles in an image is 17.4, that of pedestrian is 1.4, and that of cyclists is 0.1. It is rarely to see cyclists in the datasets, but I don't want to ignore the cyclists, because this project is for SDC, and ths system should be safe. Moreover, I can say about dataset:
- Environment: Datasets includes freeway, city, suburban, and so on. Data balance looks good.
- Weather: Datasets includes day/night/foggy/rainy scenario. It is rarely to take data in bad weather condition, but I want enough data to make model robust. 
- Classes: There are many object on road, I need not only vehicle/pedestrien/cyclist but also bus/truck/motorcycle, and so on. I also want more data about cyclist. 

### Cross validation

Cross validation is a method to separate dataset for train and evaluation, and measure model performance. This project didn't use this validation, but it is good idea to use `KFold` in `sklearn.model_selection` to execute cross validation. The following is 

## Training

I choose reference model. I have change config and train/eval. The results are saved in this file structure:

### Improve on the reference
For this project, I decided to use SSD Resnet 50 as a baseline to make it easy to
study how to improve model. 

This result of baseline is not good. The loss is greater than 10 and I want to improve this.

The following is a result of default model used as reference.
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.013
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.002
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.009
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.060
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.023
INFO:tensorflow:    + Loss/localization_loss: 0.989506
INFO:tensorflow:    + Loss/classification_loss: 0.712536
INFO:tensorflow:    + Loss/regularization_loss: 47.339748
INFO:tensorflow:    + Loss/total_loss: 49.041794
```


I checked mAP, recall, and loss. This figure shows mAP. This process was executed in workspace which Udacity provided and I wasn't able to plot mAP curve. ([Issue report](https://knowledge.udacity.com/questions/802296))

This is result of precision. The single plot is precision result at 2500 step and mAP is 0.0. 
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ref_precision.png" width="800">

This is result of recall. The single plot is recall result at 2500 step and recall is 0.0.
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ref_recall.png" width="800">

This is result of loss. This figure was able to show loss curve during train process.
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ref_loss.png" width="800">

mAP, recall, and loss of default model were not enough for SDC project. I guessed that the default model could be overffitting, because the loss metrics did not converge to small values in training proceeded. 

Fist, I tried data augmentation to improve learning efficiency and avoid overfitting. Next, I adjusted leaning rate annealing.

I choose loss as metrics to adjust model, and set target total loss 1.0 or less.

#### Apply data Augmentation

I apply data augmentation and train again. The default model uses `random_horizontal_flip` and `random_crop_image`, but this augmentation cannot modify original image and this result is as same as default images. I applied these data augments. This project uses Tf Object Detection API, I can use [preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) to add data augmentation. What I actually implemented is to modify `pipeline_new.config` file. The following is code I add to the file. I assumed that Augmentaion about color and black patch are necessary to improve the model.

```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_distort_color {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
    }
  }
  data_augmentation_options {
    random_adjust_hue {
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
    }
  }
  data_augmentation_options {
    random_black_patches{
    }
  }
  data_augmentation_options {
    random_image_scale {
      min_scale_ratio: 0.5
      max_scale_ratio: 1.5
    }
  }
```

This result is [here](https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/result.txt).

```
open jupytor notebook
- Exploratory Data Analysis.ipynb
```

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/da_example.png" width="400">

Using this data, I train SSD model and result is the following. mAP and recall results couldn't be checked using jupyter, I focussed on total loss and leeaning rate.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex1_ada/lr.png" width="400">

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
INFO:tensorflow:    + Loss/localization_loss: 1.179318
INFO:tensorflow:    + Loss/classification_loss: 0.774768
INFO:tensorflow:    + Loss/regularization_loss: 5.014042
INFO:tensorflow:    + Loss/total_loss: 6.968129
```
Even if I apply the data augmentation loss was still large. mAP and recall can be improved more.

Next, I tried change optimizer.

#### Optimizer and learning rate
The following is a result using RMS optimizer instead of Ada optimizer.
<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex2_rms/lr.png" width="400">

The detail of result is here.

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.023
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.004
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.027
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.004
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.016
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.038
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.165
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.125
INFO:tensorflow:    + Loss/localization_loss: 0.811599
INFO:tensorflow:    + Loss/classification_loss: 0.522778
INFO:tensorflow:    + Loss/regularization_loss: 0.303872
INFO:tensorflow:    + Loss/total_loss: 1.638250
```

Since learning rate is same and RMS loss gets better than Ada loss, RMS should be used than Ada optimizer.

Next I adjust parameter about learing rate with RMS.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/loss.png" width="400">

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/rl.png" width="400">

The detail of result is here.

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.056
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.114
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.051
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.228
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.250
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.013
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.061
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.303
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.319
INFO:tensorflow:Eval metrics at step 2500
INFO:tensorflow:    + Loss/localization_loss: 0.564275
INFO:tensorflow:    + Loss/classification_loss: 0.307815
INFO:tensorflow:    + Loss/regularization_loss: 0.246372
INFO:tensorflow:    + Loss/total_loss: 1.118462
 ```

To check the detail, I also share figures in Tensorboard. As I have already mentioned, I wasn't able to plot precision and recall curve during train process, the single plot is shown in figure.

This figure shows that mAP result. 

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ex3_precision.png" width="800">

The following shows recall result, but I also wasn't able to plot recall curve.

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ex3_recall.png" width="800">

It is difficult to analyze figures, but mAP of ex3 is better than ex2 at 2500 step result in text data. In terms of recall, ex3 result is better than ex2.

The following shows loss. 

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/images/ex3_loss.png" width="800">

For the training loss, both localization and classification loss can converge.

I restrict learing rate at begining of epoch, loss is restricted and it keeps around 1.0. I almost meet our goal.

Here is a list I want to try to improve object detectiong performace.
- Increase the number of data
- Increase the number of scene data types
- Increase the number of class types
- Use different optimization
- Use cross-validation
- Test other model than SSD

### Inference result

<img src="https://github.com/n-kamata/ObjectDetection/blob/master/experiments/ex3_rms_base/animation.gif" width="400">
This is a result of trained SSD. Object detecter developed by this project can work in night.