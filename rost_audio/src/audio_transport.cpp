#include <stdio.h>
#include <string>
#include <gst/gst.h>
extern "C"
{
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
}
#include <boost/thread.hpp>

#include <ros/ros.h>
#include <rost_audio/AudioRaw.h>

using namespace std;

namespace audio_transport
{
  class RosGstCapture
  {
    public: 
      RosGstCapture(ros::NodeHandle nhp)
      {
        // The bitrate at which to encode the audio
        nhp.param<int>("samplerate", samplerate, 44100);
	nhp.param<string>("audiosource", source_string, "");

        _pub = _nh.advertise<rost_audio::AudioRaw>("audio", 5, true);

        _loop = g_main_loop_new(NULL, false);
        	
        // We create the sink first, just for convenience
	_sink = gst_element_factory_make("appsink", "sink");
	g_object_set(G_OBJECT(_sink), "emit-signals", true, NULL);
	g_object_set(G_OBJECT(_sink), "max-buffers", 1000, NULL);
	g_signal_connect( G_OBJECT(_sink), "new-buffer", 
			  G_CALLBACK(onNewBuffer), this);

	GError ** error;
	char config[256];
	if (source_string.compare("") == 0 or source_string.compare("alsasrc") == 0) {
	  sprintf(config, "alsasrc ! audioconvert ! audio/x-raw-int, format=U16, rate=%d, channels=1 ! audioresample", samplerate);
	}
	else {
 	  sprintf(config, "filesrc location=%s ! decodebin ! audioconvert ! audio/x-raw-int, format=U16, rate=%d, channels=1 ! audioresample", source_string.c_str(), samplerate);
	}
	cout << "starting pipeline" << endl;
	cout << config << endl;
	_pipeline = gst_parse_launch(config, error);
	if (_pipeline == NULL)
	{
	  std::cout << "DEAD!" << std::endl;
	  exit(-1);
	}
	GstPad *outpad = gst_bin_find_unlinked_pad(GST_BIN(_pipeline), GST_PAD_SRC);
	g_assert(outpad);
	GstElement *outelement = gst_pad_get_parent_element(outpad);
	g_assert(outelement);
	gst_object_unref(outpad);
	if (!gst_bin_add(GST_BIN(_pipeline), _sink)) {
	    fprintf(stderr, "gst_bin_add() failed\n"); // TODO: do some unref
	    gst_object_unref(outelement);
	    gst_object_unref(_pipeline);
	    return;
	}

	if (!gst_element_link(outelement, _sink)){
	  fprintf(stderr, "GStreamer: cannot link outelement(\"%s\") -> sink\n", gst_element_get_name(outelement));
	  gst_object_unref(outelement);
	  gst_object_unref(_pipeline);
	  return;
	}
	gst_object_unref(outelement);

	gst_element_set_state(GST_ELEMENT(_pipeline), GST_STATE_PLAYING);

        _gst_thread = boost::thread( boost::bind(g_main_loop_run, _loop) );
      }

      void publish( const rost_audio::AudioRaw &msg )
      {
        _pub.publish(msg);
      }

      static GstFlowReturn onNewBuffer (GstAppSink *appsink, gpointer userData)
      {
        RosGstCapture *server = reinterpret_cast<RosGstCapture*>(userData);

        GstBuffer *buffer;
        g_signal_emit_by_name(appsink, "pull-buffer", &buffer);
        
        rost_audio::AudioRaw msg;
	
	
	for (int i = 0; i < buffer->size; i+=2){
	  long val = 0;
	  for (int j = 0; j < 2; j++){
	    val += (buffer->data[j + i] << (j*8));
	  }
	  cout << val << endl;
	}
	/*double out;
	int consec_zeros = 0;
	int size;
	while (size*2 < buffer->size){
	  long val = 0;
	  for (int i = 0; i < 2; i++){
	    val += (buffer->data[size + i] << ((1-i)*8));
	  }
	  if (val == (1 << 15)){
	    consec_zeros++;
	    if (consec_zeros >= 5){
	      for (int j = 0; j < 5; j++){
		msg.data.pop_back();
	      }
	      break;
	    }
	  }
	  else{
	    consec_zeros = 0;
	  }
	  out = ((double) val / (1 << 15));
	  msg.data.push_back(out);
	  size++;
	}
        msg.samplerate = server->samplerate;
        msg.size = size;*/

        server->publish(msg);

        return GST_FLOW_OK;
      }

    public:
      int samplerate;
      int precision;
      string source_string;
    private:
      ros::NodeHandle _nh;
      ros::Publisher _pub;

      boost::thread _gst_thread;

      GstElement *_pipeline, *_sink;
      GMainLoop *_loop;
  };
}

int main (int argc, char **argv)
{
  if (argv[1] != NULL && strcmp(argv[1], "--help") == 0){
    cout << endl;
    cout << "Usage: rosrun rost_audio audiotransport" << endl;
    cout << "Parameters: " << endl;
    cout << "    _samplerate (int), default 44100" << endl;
    cout << "    _audiosource (string), default, alsasrc, if otherwise specifies an audio file" << endl;
    cout << endl;
    return -1;
  }
  ros::init(argc, argv, "audio_capture");
  gst_init(&argc, &argv);
  
  ros::NodeHandle nhp("~");
  audio_transport::RosGstCapture server(nhp);
  ros::spin();
}
