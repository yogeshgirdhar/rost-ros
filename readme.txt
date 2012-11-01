Requirements:
Ubuntu 12.04 or above with ROS Fuerte and OpenCV installed


Getting the source:
1) hg clone ssh://username@moist.cim.mcgill.ca//home/discovery/mrl/hgrepo/girdhar/ros/rost
2) hg clone ssh://username@moist.cim.mcgill.ca//home/discovery/mrl/hgrepo/ros/cv_image_source

Building:
3) rosmake cv_image_source
4) rosmake rost_common rost_topics rost_vision rost_visualize

Running on camera:
roslaunch rost_launch camera.launch [options]

Running a video file:
roslaunch rost_launch file.launch file:=/absolute/path/of/video.mp4 [options]


Optional arguments the the launch file(with default values shown):
alpha:=0.1       controls the smoothness of topic distribution describing an image patch
beta:=0.1        controls the smoothness of word distribution describing a topic
tau:=2.0         how important is present compared to past for topic refinement. 
                 1.0 => all times are equally important
                 tau > 1 => present is more important
                 0<tau<1 => past is more important 
K:=16            number of topics
S:=16            summary size
vout:=<out.avi>  record the visualization video. output is MJPEG avi file.
rate:=5          process video at 5 fps
subsample:=2     temporally subsample the video (only applicable when using file.launch)
loop:=false      loop the video (only with file.launch)
scale:=1.0       scale the input video. To process the video at half the linear size, set scale:=0.5
threads:=4       number of threads to spawn for topic refinement. 
cell.width:=160  split the input image into windows of size 160x160


There are many more options, but you probably don't care about them. Let me know if you have any questions
