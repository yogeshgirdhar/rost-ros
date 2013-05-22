#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <sndfile.h>
#include <ros/ros.h>
#include <gst/gst.h>
#include "rost_audio/AudioRaw.h"

using namespace std;

SNDFILE *sf;
SF_INFO *info;

void audioCallback(const rost_audio::AudioRawConstPtr &msg){
  int consec_zeros = 0;
  int size = 0;
  while (size < msg->data.size()){
    if (msg->data[size] == 0){
      consec_zeros++;
      if (consec_zeros >= 5){
        size -= 5;
        break;
      }
    }
    else {
      consec_zeros = 0;
    }
    size++;
  }
  double * sample = (double *) malloc(sizeof(double) * size);
  for (int i = 0; i < msg->data.size(); i++){
    sample[i] = ((double) msg->data[i]);
  }

  // write the entire buffer to the file
  sf_count_t count = sf_write_double ( sf, &sample[0], msg->data.size() );

  // force write to disk
  sf_write_sync( sf );
}

int main(int argc, char *argv[]){
  if (argv[1] != NULL && strcmp(argv[1], "--help") == 0){
    cout << endl;
    cout << "Usage: rosrun rost_audio save_wav" << endl;
    cout << "Parameters:" << endl;
    cout << "    _outfile (string), default out.wav" << endl;
    cout << endl;
    return -1;
  }
  ros::init(argc, argv, "audio_words");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  
  string outfile_name;
  nhp.param<string>("outfile", outfile_name, "out.wav");
  cout << outfile_name << ": name" << endl;

  info = (SF_INFO *) malloc(sizeof(SF_INFO));
  info->channels = 1;
  info->samplerate = 44100;
  info->format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  sf = sf_open( outfile_name.c_str(), SFM_WRITE, info );  
  
  ros::Subscriber audio_sub = nh.subscribe("audio", 1, audioCallback);

  ros::spin();
  sf_close(sf);
}
