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
OpenCV
```


