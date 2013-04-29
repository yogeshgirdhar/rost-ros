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
      RosGstCapture()
      {
        // The bitrate at which to encode the audio
	ros::NodeHandle _nhp("~");
        _nhp.param<int>("samplerate", samplerate, 44100);
	_nhp.param<string>("audiosource", source_string, "");

        _pub = _nh.advertise<rost_audio::AudioRaw>("audio", 10, true);

        _loop = g_main_loop_new(NULL, false);
        _pipeline = gst_pipeline_new("ros_pipeline");

	_caps = gst_caps_new_simple("audio/x-raw-float", "rate", G_TYPE_INT, samplerate, "channels", G_TYPE_INT, 1, "width", G_TYPE_INT, 32, NULL);
	char * caps_s;
	caps_s = gst_caps_to_string(_caps);
	std::cout << caps_s << std::endl;
	
        // We create the sink first, just for convenience
	_sink = gst_element_factory_make("appsink", "sink");
	g_object_set(G_OBJECT(_sink), "emit-signals", true, NULL);
	g_object_set(G_OBJECT(_sink), "max-buffers", 100, NULL);
	g_signal_connect( G_OBJECT(_sink), "new-buffer", 
			  G_CALLBACK(onNewBuffer), this);

	if (source_string.compare("") == 0 or source_string.compare("alsasrc") == 0) {
	  _source = gst_element_factory_make("alsasrc", "source");
	}
	else {
	  _source = gst_element_factory_make("filesrc", "source");
	  g_object_set(G_OBJECT(_source), "location", source_string.c_str(), NULL);
	  cout << "Using " << source_string << " as audio source" << endl;
	}
        _convert = gst_element_factory_make("audioconvert", "convert");

        gst_bin_add_many( GST_BIN(_pipeline), _source, _convert, _sink, NULL );

	bool success = gst_element_link_filtered(_convert, _sink, _caps);
	gst_caps_unref(_caps);
	if (success) std::cout << "caps worked" << std::endl;
	else std::cout << "caps failed" << std::endl;

        success = gst_element_link(_source, _convert);
	if (success) std::cout << "link worked" << std::endl;
	else std::cout << "link failed" << std::endl;

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
        msg.data.resize( buffer->size );
        memcpy( &msg.data[0], buffer->data, buffer->size);
        msg.samplerate = server->samplerate;
        msg.size = buffer->size;

        server->publish(msg);

        return GST_FLOW_OK;
      }

    public:
      int samplerate;
      string source_string;
    private:
      ros::NodeHandle _nh;
      ros::Publisher _pub;

      boost::thread _gst_thread;

      GstElement *_pipeline, *_source, *_sink, *_convert;
      GstCaps *_caps;
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

  audio_transport::RosGstCapture server;
  ros::spin();
}
