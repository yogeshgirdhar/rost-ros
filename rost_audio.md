## Getting Started ##

  1. Install ROST as per [Getting Started](http://code.google.com/p/rost-ros/wiki/GettingStarted)
  1. Generate a single channel, 44.1 kHz sample rate, uncompressed wav file of the sound to be modelled.
  1. Run the audiowords.launch file with the command
> > rosrun audio\_words audiowords.launch file:=<the wav file>

> This will generate audio words for the wav file based on default parameters and the supplied MontrealSounds2k.txt vocabulary. Words are published to the /words topic. /words/words is a list of labels for the current observation and /words/word\_pose is the time in ms of the time window corresponding to each word.

## Generating and Using Your Own Vocabulary ##

The provided MontrealSound2k.txt vocabulary is a 2000 word vocabulary generated from assorted ambient sounds recorded around Montreal and accessible at [montrealsoundmap.com](http://www.montrealsoundmap.com/). It is a reasonable choice for modelling ambient city sounds, but you may decide to generate your own vocabulary to better suit your own purposes.

Start by preparing one or many single channel, 44.1 kHz sample rate, uncompressed wav files of the data you wish to be used in the vocabulary. The gen\_vocab program  takes a list of these files, computes MFCCs for each, quantizes them using k-means, and produces a text file containing the vocabulary. Command line arguments are as follows:

  * --help                     produce help message.
  * --wav\_names arg            A list of the wav files to use as source audio for the vocabulary. Must be specified.
  * --vocab\_name arg           The text file to save the vocab to.
  * --vocab\_size arg (=2000)   Number of distinct words in the output vocabulary
  * --fft\_buf\_size arg (=4096) Number of samples taken into account when calculating the fft and mfcc.
  * --overlap arg (=0.5)       Amount of overlap between successive mfccs. Must be < 1