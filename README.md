# **PupilDetector** #

This project is a different version of the DeepVOG model, presented in the credits section below.

The original DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network.
Currently, it is available for offline gaze estimation of eye-tracking video clips.

In this work, we used the DeepVOG model to predict the pupil center coordinates of a given eye image and NOT a video as in the original DeepVOG project. Also, we present the capability to respond to HTTP requests that contain input images, by sending the pupil center coordinates back.

## Installation ##

**1. By a docker image**
```ruby
$
```

**2. Package installation**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
2.1 Please pull the package from repository:
```ruby
$
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
2.2 Please move to the PupilDetector directory and type: 
```ruby
$ python setup.py install
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
2.3 Please install the following prerequisites: 
```ruby
numpy
scikit-video
scikit-image
tensorflow-gpu
keras
flask
opencv
```

## Usage (Command-line interface) ##

In order to use the DeepVOG model, it is necessary to first fit an eyeball model on a given input video. There is an example of that as a JSON file available in the repository. The deserved command is:
```ruby
$ python -m deepvog --fit [VIDEO_FIT_PATH] [EYEBALL_MODEL_PATH]
```

If there is no need to fit an eyeball model as described, there are two ways to predict the pupil center:
1. By the "infer" mode:
```ruby
$ python -m deepvog --infer [IMAGE_PATH] [EXISTING_EYEBALL_MODEL_PATH] [CSV_RESULTS_FILE_PATH] -v [IMAGE_INFERENCE_PATH]
```

2. By an HTTP request:
```ruby
$ curl -X POST -F "image_file=@exmp.jpg" http://$PUBLIC_IP:5000/
```

## Credits ##
For more information, visit the original DeepVOG repository in the following link: https://github.com/pydsgz/DeepVOG

<p align="center">
  <img width="240" height="190" src="deepvog_exmp.gif">
</p>
