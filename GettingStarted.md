# Introduction #

Add your content here.


# Requirements #
Ubuntu 12.04 or above with ROS Groovy

## Dependencies ##
  * **cv\_image\_source** : ROST uses cv\_image\_source ROS package by default for getting images from camera or video files. To install it:
    * `$ hg clone https://bitbucket.org/yogeshgirdhar/cv_image_source`
    * `$ rosmake cv_image_source`
  * **system dependencies** :
    * `$ sudo apt-get install libctemplate-dev libsndfile1-dev libyaml-cpp0.3-dev libpulse-dev libfftw3-dev`


# Building ROST #
  * `$ hg clone https://code.google.com/p/rost-ros/`
  * `$ rosmake rost-ros`

# Using ROST #

### Using ROST with camera ###
`$ roslaunch rost_launch rost.launch camera:=true [options]`

### Using ROST with a video file ###
`$ roslaunch rost_launch rost.launch file:=/absolute/path/of/video.mp4 [options]`


Optional arguments the the launch file(with default values shown):
  * alpha:=0.1       controls the smoothness of topic distribution describing an image patch
  * beta:=1.0        controls the smoothness of word distribution describing a topic
  * tau:=2.0         how important is present compared to past for topic refinement.
    1. 0 => all times are equally important
> > tau > 1 => present is more important
> > 0<tau<1 => past is more important
  * K:=16            number of topics
  * S:=16            summary size
  * vout:=<out.avi>  record the visualization video. output is MJPEG avi file.
  * video.rate:=5          process video at 5 fps
  * video.subsample:=2     temporally subsample the video (only applicable when using file.launch)
  * video.loop:=false      loop the video (only with file.launch)
  * video.scale:=1.0       scale the input video. To process the video at half the linear size, set scale:=0.5
  * threads:=4       number of threads to spawn for topic refinement.
  * cell.width:=160  split the input image into windows of size 160x160
  * image.width:=640       tell the system the size of the input vide
  * image.height:=480

There are many more options. To see them open the launch file.