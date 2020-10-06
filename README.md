# **PupilDetector** #

This project is a different version of the DeepVOG model, presented in the credits section below.

The original DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network.
Currently, it is available for offline gaze estimation of eye-tracking video clips.

In this work, we used the DeepVOG model to predict the pupil center coordinates of a given eye image and NOT a video as in the original DeepVOG project. Also, we allow this prediction by handling HTTP requests that contain input eye images.

## Installation by a docker image ##

Pull the docker image to run the Flask server:
```ruby
$ docker pull 247555/pupildetector:latest
```

## Usage (Command-line interface) ##

**Note:** The following 'fit' and the 'infer' commands are only available through the DeepVOG repository, mentioned in the credits section. Also, these functionalities don't depend on running the Flask server as described in 2. below.

In order to use the DeepVOG model, it is necessary to first fit an eyeball model on a given input video. There is an example of that as a JSON file available in the repository. The deserved command is:
```ruby
$ python -m deepvog --fit [VIDEO_FIT_PATH] [EYEBALL_MODEL_PATH]
```

If there is no need to fit an eyeball model as described, there are two ways to predict the pupil center:
1. By the "infer" mode:
```ruby
$ python -m deepvog --infer [IMAGE_PATH] [EXISTING_EYEBALL_MODEL_PATH] [CSV_RESULTS_FILE_PATH] -v [IMAGE_INFERENCE_PATH]
```

2. By an HTTP request
2.1 Run the Flask server through the container:
```ruby
$ docker run -d --runtime=nvidia -p 5000:5000 247555/pupildetector:latest python /notebooks/DeepVOG_dspip/deepvog/_main_.py
```

2.2 Send the HTTP request:
```ruby
$ curl -X POST -F "image_file=@exmp.jpg" http://$PUBLIC_IP:5000/
```

## Credits ##
For more information, visit the original DeepVOG repository in the following link: https://github.com/pydsgz/DeepVOG

<p align="center">
  <img width="240" height="190" src="deepvog_exmp.gif">
</p>
