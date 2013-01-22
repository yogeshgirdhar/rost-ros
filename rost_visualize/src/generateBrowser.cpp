#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include "../yaml-cpp/include/yaml-cpp/yaml.h"

namespace enc = sensor_msgs::image_encodings;

using namespace std;

typedef map<int, map<int, int> >::iterator int_mapintint_it;
typedef map<int, map<int, float> >::iterator int_mapintfloat_it;
typedef map<int, int>::iterator intint_it;
typedef map<int, float>::iterator intfloat_it;

bool copyDir(boost::filesystem::path const & source, boost::filesystem::path const & destination){
    namespace fs = boost::filesystem;
    try{// Check whether the function call is valid
        if(!fs::exists(source) || !fs::is_directory(source)){
            std::cerr << "Source directory " << source.string()
                << " does not exist or is not a directory." << std::endl;
            return false;
        }
        if(fs::exists(destination)){
            std::cerr << "Destination directory " << destination.string()
                << " already exists." << std::endl;
            return false;
        }
        // Create the destination directory
        if(!fs::create_directory(destination)){
            std::cerr << "Unable to create destination directory" << destination.string() << std::endl;
            return false;
        }
    }
    catch(fs::filesystem_error const & e){
        std::cerr << e.what() << std::endl;
        return false;
    }
    // Iterate through the source directory
    for(fs::directory_iterator file(source);
        file != fs::directory_iterator(); ++file){
        try{
            fs::path current(file->path());
            if(fs::is_directory(current)){
                // Found directory: Recursion
                if(!copyDir(current, destination / current.filename())){
                    return false;
                }
            }
            else{                // Found file: Copy
                fs::copy_file(current, destination / current.filename());
            }
        }
        catch(fs::filesystem_error const & e){
            std:: cerr << e.what() << std::endl;
        }
    }
    return true;
}

map<int, int> topicsHistogram(vector<int> topics){
    map<int, int> histogram;
    for (unsigned int i = 0; i < topics.size(); i++){
        int topic = topics[i];
        if (histogram.count(topic) > 0){
            histogram[topic]++;
        }
        else {
            histogram[topic] = 1;
        }
    }
    return histogram;
}

bool mapHasCloseSeq(map<int, int> best_seq_occ, int newSeq, int min_dist){
    for (intint_it seq_occ = best_seq_occ.begin(); seq_occ != best_seq_occ.end(); seq_occ++){
        if (newSeq > seq_occ->first && newSeq-seq_occ->first < min_dist){
            return true;
        }
        if (newSeq < seq_occ->first && seq_occ->first - newSeq < min_dist){
            return true;
        }
    }
    return false;
}

map<int, int> getMostLikelySeqs(int k, unsigned int n, map<int, map <int, int> > histograms){
    // seq, occurrences
    map<int, int> best_seq_occ;
    // for each histogram
    for(int_mapintint_it seq_hist = histograms.begin(); seq_hist != histograms.end(); seq_hist++) {
        // if the topic in question is there
        if (seq_hist->second.count(k) > 0){
            // if we are just starting
            if (best_seq_occ.empty()){
                // insert whatever
                best_seq_occ.insert(best_seq_occ.begin(), pair<int, int>((seq_hist->first), seq_hist->second[k]));
                continue;
            }
            // otherwise, for each of the most likely values
            for (intint_it seq_occ = best_seq_occ.begin(); seq_occ != best_seq_occ.end(); seq_occ++){
                // if the value in the histogram is more than the mostLikely value
                if (seq_hist->second[k] > seq_occ->second && !mapHasCloseSeq(best_seq_occ, seq_hist->first, 8)){
                    // insert its sequence before this mostLikely value
                    best_seq_occ.insert(seq_occ, pair<int, int>(seq_hist->first, seq_hist->second[k]));
                    // if there are more mostLikely's than we wanted
                    while (best_seq_occ.size() > n){
                        // erase the smallest one
                        best_seq_occ.erase(seq_occ->first);
                    }
                    break;
                }
            }
        }
    }
    return best_seq_occ;
}

void saveImage(sensor_msgs::Image::ConstPtr msg, int topic, int seq, char *path){
    cv_bridge::CvImagePtr cv_ptr;
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    char fname[60];
    sprintf(fname, "%stop%d-seq%d.png", path, topic, seq);
    imwrite(fname, cv_ptr->image);
}

void saveImages(string bagname, string topicname, vector<map<int, int> > mostLikely, char * path){
    rosbag::Bag bag(bagname);
    rosbag::View view(bag, rosbag::TopicQuery(topicname));
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::Image::ConstPtr i = m.instantiate<sensor_msgs::Image>();
        if (i != NULL){
            for (unsigned int topic = 0; topic < mostLikely.size(); topic++){
                if (mostLikely[topic].count(i->header.seq) > 0){
                    //cout << topic << ",";
                    saveImage(i, topic, i->header.seq, path);
                }
            }
        }
    }
    bag.close();
}

void saveAllImages(string bagname, string topicname, char * path, int * minSeq, int * maxSeq){
    rosbag::Bag bag(bagname);
    rosbag::View view(bag, rosbag::TopicQuery(topicname));
    BOOST_FOREACH(rosbag::MessageInstance const m, view){
        sensor_msgs::Image::ConstPtr i = m.instantiate<sensor_msgs::Image>();
        if (i != NULL){
            char fname[60];
            if (i->header.seq < *minSeq) *minSeq = i->header.seq;
            if (i->header.seq > *maxSeq) *maxSeq = i->header.seq;
            if (i->header.seq % 20 == 0) cout << "..seq: " << i->header.seq << endl;
            sprintf(fname, "%s%d.png", path, i->header.seq);
            imwrite(fname, (cv_bridge::toCvCopy(i, enc::BGR8))->image);
        }
    }
    bag.close();
}

map<int, int> calcOverallHist(map<int, map <int, int> > histograms){
    map<int, int> overallHist;
    for(int_mapintint_it seq_hist = histograms.begin(); seq_hist != histograms.end(); seq_hist++) {
        for (intint_it top_occ = seq_hist->second.begin(); top_occ != seq_hist->second.end(); top_occ++){
            if (overallHist.count(top_occ->first) > 0){
                overallHist[top_occ->first] += top_occ->second;
            }
            else{
                overallHist[top_occ->first] = top_occ->second;
            }
        }
    }
    return overallHist;
}

map<int, float> normalizeHistogram(map<int, int> hist){
    map<int, float> normaled;
    int total = 0;
    for (intint_it top_occ = hist.begin(); top_occ != hist.end(); top_occ++){
        total += top_occ->second;
    }
    for (intint_it top_occ = hist.begin(); top_occ != hist.end(); top_occ++){
        normaled[top_occ->first] = (float) top_occ->second/total;
    }
    return normaled;
}

map<int, map<int, float> > normalizeHistograms(map<int, map<int, int> > histograms){
    map<int, map<int, float> > normaled;
    for(int_mapintint_it seq_hist = histograms.begin(); seq_hist != histograms.end(); seq_hist++) {
        normaled[seq_hist->first] = normalizeHistogram(seq_hist->second);
    }
    return normaled;
}

void writeTopicPage(int topic, map<int, int> mostLikely, map<int, int> overallHist, map<int, map<int, float> > histograms, char *path){
    char page_name[90];
    sprintf(page_name, "%stopic_%d.html", path, topic);
    cout << "Writing topic " << topic << " in " << page_name << endl;
    FILE * page = fopen(page_name, "w");
    fprintf(page, "<!doctype html>\n");
    fprintf(page, "\t<html>\n");
    fprintf(page, "\t\t<head>\n");
    fprintf(page, "\t\t\t<script type=\"text/javascript\" src=\"https://www.google.com/jsapi\"></script>\n");
    fprintf(page, "\t\t\t<script type=\"text/javascript\">\n");
    fprintf(page, "\t\t\t\tgoogle.load(\"visualization\", \"1\", {packages:[\"corechart\"]});\n");
    fprintf(page, "\t\t\t\tgoogle.setOnLoadCallback(drawChart);\n");
    fprintf(page, "\t\t\t\t\tfunction drawChart() {\n");
    fprintf(page, "\t\t\t\t\tvar data = google.visualization.arrayToDataTable([\n");
    fprintf(page, "\t\t\t\t\t['Topic', 'Weight', 'This Topic Weight'],\n");
    for (int i = 0; i < 64; i++){
        fprintf(page, "\t\t\t\t\t['");
        if ((i % 10) == 0) fprintf(page, "%d", i);
        if (i == topic) fprintf(page, "', 0, %d],\n", overallHist[i]);
        else fprintf(page, "', %d, 0],\n", overallHist[i]);
    }
    fprintf(page, "\t\t\t\t\t\t]);\n");
    fprintf(page, "\t\t\t\t\tvar options = {\n");
    fprintf(page, "\t\t\t\t\tcht: 'bvs',\n");
    fprintf(page, "\t\t\t\t\ttitle: 'Topic Weights',\n");
    fprintf(page, "\t\t\t\t\tlegend: {position: 'none'},\n");
    fprintf(page, "\t\t\t\t\tchartArea: {left:40,top:40,width:\"100%\",height:\"80%\"},\n");
    fprintf(page, "\t\t\t\t\tvAxis: {title: 'Topic',  titleTextStyle: {color: 'red'}},\n");
    fprintf(page, "\t\t\t\t\thAxis: {title: 'Occurrences',  titleTextStyle: {color: 'red'}}\n");
    fprintf(page, "\t\t\t\t\t};\n\n");
    fprintf(page, "\t\t\t\t\tvar chart = new google.visualization.BarChart(document.getElementById('chart_div'));\n");
    fprintf(page, "\t\t\t\t\tchart.draw(data, options);\n");

    fprintf(page, "\t\t\t\tvar handler = function(e) {\n");
    fprintf(page, "\t\t\t\t\tvar sel = chart.getSelection();\n");
    fprintf(page, "\t\t\t\t\tvar row = sel[0]['row'];\n");
    fprintf(page, "\t\t\t\t\tvar url = \"./topic_\" + row + \".html\";\n");
    fprintf(page, "\t\t\t\t\twindow.location = url\n");
    fprintf(page, "\t\t\t\t}\n");
    fprintf(page, "\t\t\tgoogle.visualization.events.addListener(chart, 'select', handler);\n");

    fprintf(page, "\t\t\t\t\t}\n");
    fprintf(page, "\t\t\t\t\t</script>\n");

    fprintf(page, "\t\t<meta charset=\"utf-8\">\n");
    fprintf(page, "\t\t<meta http-equiv=\"X-UA-Compatible\" content=\"chrome=1\">\n");
    fprintf(page, "\t\t\t<title>ROST Topic Browser</title>\n\n");
    fprintf(page, "\t\t\t<link rel=\"stylesheet\" href=\"../resource/stylesheets/styles.css\">\n");
    fprintf(page, "\t\t\t<link rel=\"stylesheet\" href=\"../resource/stylesheets/pygment_trac.css\">\n");
    fprintf(page, "\t\t\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, user-scalable=no\">\n");
    fprintf(page, "\t\t</head>\n");
    fprintf(page, "\t\t\t<body>\n");
    fprintf(page, "\t\t\t<div class=\"wrapper\">\n");
    fprintf(page, "\t\t\t<header>\n");
    fprintf(page, "\t\t\t<a href=\"../all-topics.html\"><h1>ROST Visual Topic Model Browser</h1></a>\n");
    fprintf(page, "\t\t\t<div id=\"chart_div\" style=\"width: 325px; height: 500px;\"></div>\n\n");
    fprintf(page, "\t\t\t</header>\n\n");
    fprintf(page, "\t\t\t<section>\n");
    fprintf(page, "\t\t\t<p><h3>Topic %d Maximum Likelihood Images:</h3></p>\n", topic);
    for (intint_it seq_occ = mostLikely.begin(); seq_occ != mostLikely.end(); seq_occ++){
        fprintf(page, "\t\t\tSeq %d:\n", seq_occ->first);
        fprintf(page, "\t\t\t\t<p><a href=\"../all-images.html#%d\"><img src=\"../data/best/top%d-seq%d.png\"", seq_occ->first, topic, seq_occ->first);
        fprintf(page, "title=\"Seq. %d, %d Occurrences of Topic %d\"></a>\n", seq_occ->first, seq_occ->second, topic);
        fprintf(page, "\t\t\t\tAlso in this image:\n");
        map<int, float> hist = histograms[seq_occ->first];
        for (intfloat_it top_pct = hist.begin(); top_pct != hist.end(); top_pct++){
            if (top_pct->second > 0.1){
                fprintf(page, "\t\t\t\t<a href=\"./topic_%d.html\">Topic %d (%2.f \%)</a>,\n", top_pct->first, top_pct->first, top_pct->second*100.0);
            }
        }
        fprintf(page, "</p>\n");
    }
    fprintf(page, "\t\t\t</section>\n");
    fprintf(page, "\t\t\t</div>\n");
    fprintf(page, "\t\t\t<script src=\"../resource/javascripts/scale.fix.js\"></script>\n");
    fprintf(page, "\t\t</body>\n");
    fprintf(page, "\t</html>\n");
    fclose(page);
}

void writeAllTopicsPage(int K, vector<map<int, int> > mostLikely, map<int, int> overallHist, char *path){
    char page_name[90];
    sprintf(page_name, "%sall-topics.html", path);
    FILE * page = fopen(page_name, "w");
    fprintf(page, "<!doctype html>\n");
    fprintf(page, "\t<html>\n");
    fprintf(page, "\t\t<head>\n");
    fprintf(page, "\t\t\t<script type=\"text/javascript\" src=\"https://www.google.com/jsapi\"></script>\n");
    fprintf(page, "\t\t\t<script type=\"text/javascript\">\n");
    fprintf(page, "\t\t\t\tgoogle.load(\"visualization\", \"1\", {packages:[\"corechart\"]});\n");
    fprintf(page, "\t\t\t\tgoogle.setOnLoadCallback(drawChart);\n");
    fprintf(page, "\t\t\t\t\tfunction drawChart() {\n");
    fprintf(page, "\t\t\t\t\tvar data = google.visualization.arrayToDataTable([\n");
    fprintf(page, "\t\t\t\t\t['Topic', 'Weight', 'This Topic Weight'],\n");
    for (unsigned int i = 0; i < 64; i++){
        fprintf(page, "\t\t\t\t\t['");
        if ((i % 10) == 0) fprintf(page, "%d", i);
        fprintf(page, "', %d, 0],\n", overallHist[i]);
    }
    fprintf(page, "\t\t\t\t\t\t]);\n");
    fprintf(page, "\t\t\t\t\tvar options = {\n");
    fprintf(page, "\t\t\t\t\tcht: 'bvs',\n");
    fprintf(page, "\t\t\t\t\ttitle: 'Topic Weights',\n");
    fprintf(page, "\t\t\t\t\tlegend: {position: 'none'},\n");
    fprintf(page, "\t\t\t\t\tchartArea: {left:40,top:40,width:\"100%\",height:\"80%\"},\n");
    fprintf(page, "\t\t\t\t\tvAxis: {title: 'Topic',  titleTextStyle: {color: 'red'}},\n");
    fprintf(page, "\t\t\t\t\thAxis: {title: 'Occurrences',  titleTextStyle: {color: 'red'}}\n");
    fprintf(page, "\t\t\t\t\t};\n\n");
    fprintf(page, "\t\t\t\t\tvar chart = new google.visualization.BarChart(document.getElementById('chart_div'));\n");
    fprintf(page, "\t\t\t\t\tchart.draw(data, options);\n");

    fprintf(page, "\t\t\t\tvar handler = function(e) {\n");
    fprintf(page, "\t\t\t\t\tvar sel = chart.getSelection();\n");
    fprintf(page, "\t\t\t\t\tvar row = sel[0]['row'];\n");

    // Note that this depends on the location of the topics pages!
    fprintf(page, "\t\t\t\t\tvar url = \"topics/topic_\" + row + \".html\";\n");

    fprintf(page, "\t\t\t\t\twindow.location = url\n");
    fprintf(page, "\t\t\t\t}\n");
    fprintf(page, "\t\t\tgoogle.visualization.events.addListener(chart, 'select', handler);\n");

    fprintf(page, "\t\t\t\t\t}\n");
    fprintf(page, "\t\t\t\t\t</script>\n");

    fprintf(page, "\t\t<meta charset=\"utf-8\">\n");
    fprintf(page, "\t\t<meta http-equiv=\"X-UA-Compatible\" content=\"chrome=1\">\n");
    fprintf(page, "\t\t\t<title>ROST Topic Browser</title>\n\n");
    fprintf(page, "\t\t\t<link rel=\"stylesheet\" href=\"./resource/stylesheets/styles.css\">\n");
    fprintf(page, "\t\t\t<link rel=\"stylesheet\" href=\"./resource/stylesheets/pygment_trac.css\">\n");
    fprintf(page, "\t\t\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, user-scalable=no\">\n");
    fprintf(page, "\t\t</head>\n");
    fprintf(page, "\t\t\t<body>\n");
    fprintf(page, "\t\t\t<div class=\"wrapper\">\n");
    fprintf(page, "\t\t\t<header>\n");
    fprintf(page, "\t\t\t<h1>ROST Visual Topic Model Browser</h1>\n");
    fprintf(page, "\t\t\t<div id=\"chart_div\" style=\"width: 325px; height: 500px;\"></div>\n\n");
    fprintf(page, "\t\t\t</header>\n\n");
    fprintf(page, "\t\t\t<section>\n");
    fprintf(page, "\t\t\t<p><h3>All Topics Single Maximum Likelihood Image</h3></p>\n");
    for (unsigned int i = 0; i < K; i++){
        int most = 0;
        int most_seq = 0;
        for (intint_it seq_occ = mostLikely[i].begin(); seq_occ != mostLikely[i].end(); seq_occ++){
            if (seq_occ->second > most){
                most = seq_occ->second;
                most_seq = seq_occ->first;
            }
        }
        fprintf(page, "\t\t\t\t<a href=\"topics/topic_%d.html\">Topic %d</a>\n", i, i);
        fprintf(page, "<p><a href=\"./all-images.html#%d\"><img src=\"./data/best/top%d-seq%d.png\"", most_seq, i, most_seq);
        fprintf(page, "title=\"Seq. %d\"></a></p>\n", most_seq);
    }
    fclose(page);

}

void writeAllImagesPage(char * path, map<int, map<int, float> > histograms, int minSeq, int maxSeq){
    char page_name[60];
    sprintf(page_name, "%sall-images.html", path);
    FILE * page = fopen(page_name, "w");
    fprintf(page, "<!doctype html>\n");
    fprintf(page, "<html>\n");
    fprintf(page, "<head>\t\n");
    fprintf(page, "\t<meta charset=\"utf-8\">\n");
    fprintf(page, "\t<meta http-equiv=\"X-UA-Compatible\" content=\"chrome=1\">\n");
    fprintf(page, "\t\t<title>ROST Topic Browser</title>\n\n");

    fprintf(page, "\t\t<link rel=\"stylesheet\" href=\"./resource/stylesheets/styles.css\">\n");
    fprintf(page, "\t\t<link rel=\"stylesheet\" href=\"./resource/stylesheets/pygment_trac.css\">\n");
    fprintf(page, "\t\t<link rel=\"stylesheet\" type=\"text/css\" href=\"./resource/slider/slider.css\" />\n\n");

    fprintf(page, "\t\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, user-scalable=no\">\n\n");

    fprintf(page, "\t\t<style type=\"text/css\">\n");
    fprintf(page, "\t\t\t.slider {\n");
    fprintf(page, "\t\t\twidth: 640px;\n");
    fprintf(page, "\t\t\tfont-family: Helvetica, Arial, sans-serif;\n");
    fprintf(page, "\t\t\tfont-size: 12px;\n");
    fprintf(page, "\t\t\t}\n");
    fprintf(page, "\t\t</style>\n\n");
    fprintf(page, "\t\t<script type=\"text/javascript\" src=\"./resource/slider/slider.js\"></script>\n");
    fprintf(page, "\t\t<script type=\"text/javascript\">\n");
    fprintf(page, "\t\tvar im = new Image;\n");
    fprintf(page, "\t\tvar hist_data = new Array();\n");
    for (unsigned int i = 0; i < maxSeq - minSeq; i++){
        fprintf(page, "\t\thist_data[%d] = [", i);
        if (histograms.count(i + minSeq) > 0){
            map<int, float> hist = histograms[i + minSeq];
            for (intfloat_it top_pct = hist.begin(); top_pct != hist.end(); top_pct++){
                if (top_pct->second > 0.1){
                    fprintf(page, "[%d, %2.f], ", top_pct->first, top_pct->second*100);
                }
            }
        }
        fprintf(page, "];\n");
    }
    fprintf(page, "\n");
    fprintf(page, "\t\t\twindow.onload = function(){\n");
    fprintf(page, "\t\t\tvar location = (parseInt(window.location.hash.split('#')[1]) - %d)/%d;\n", minSeq, (maxSeq - minSeq));
    fprintf(page, "\t\t\tif (isNaN(location) || location < 0) location = 0;\n");
    fprintf(page, "\t\t\tif (location > 1) location = 1;\n");
    fprintf(page, "\t\t\tslider = new Slider('my-slider',\n");
    fprintf(page, "\t\t\t{\n");
    fprintf(page, "\t\t\tsteps: %d,\n", (maxSeq - minSeq));
    fprintf(page, "\t\t\tvalue: location,\n");
    fprintf(page, "\t\t\tsnapping: true,\n");
    fprintf(page, "\t\t\tcallback: function(value) {\n");
    fprintf(page, "\t\t\t\tvar seq = Math.round(%d + value * %d);\n", minSeq, (maxSeq-minSeq));
    fprintf(page, "\t\t\t\tvar index = seq - %d;\n", minSeq);
    fprintf(page, "\t\t\t\tvar imgurl = \"./data/all/\" + seq + \".png\";\n");
    fprintf(page, "\t\t\t\tvar inhtml = \"\";\n");
    fprintf(page, "\t\t\t\tif (hist_data[index] && hist_data[index].length > 0){\n");
    fprintf(page, "\t\t\t\tfor (var i=0; i<hist_data[index].length;i++){\n\n\n\n");
    fprintf(page, "\t\t\t\tinhtml += \"<a href=\\");
    fprintf(page, "./topics/topic_\" + hist_data[index][i][0] + \".html\\");
    fprintf(page, "> Topic \" + hist_data[index][i][0];\n");
    fprintf(page, "\t\t\t\tinhtml += \" (\" + hist_data[index][i][1] +\"%) </a>\";\n");
    fprintf(page, "\t\t\t\t}\n");
    fprintf(page, "\t\t\t\t}\n");
    fprintf(page, "\t\t\t\telse{ inhtml = \"No histogram data\"; }\n");
    fprintf(page, "\t\t\t\tim.src = imgurl;\n");
    fprintf(page, "\t\t\t\tdocument.frame.src = imgurl;\n");
    fprintf(page, "\t\t\t\tdocument.frame.title = seq;\n");
    fprintf(page, "\t\t\t\tdocument.getElementById(\"seq_no\").innerHTML = \"Seq. No: \" + seq + \" - \" + inhtml;\n");
    fprintf(page, "\t\t\t}\n");
    fprintf(page, "\t\t\t});\n");
    fprintf(page, "\t\tslider.callback(location)\n");
    fprintf(page, "\t\t}\n");
    fprintf(page, "\t</script>\n");
    fprintf(page, "</head>\n");
    fprintf(page, "<body>\n");
    fprintf(page, "\t<div class=\"wrapper\">");
    fprintf(page, "\t<header>");
    fprintf(page, "\t\t<h1>ROST Visual Topic Model Browser</h1>\n");
    fprintf(page, "\t</header>\n");
    fprintf(page, "\t\t<br> <br> <br>\n");
    fprintf(page, "\t\t<p><h3>All Images</h3></p>\n");
    fprintf(page, "\t\t<div id=\"seq_no\">Seq. No: 4454</div>\n");
    fprintf(page, "\t\t<p><img src=\"./data/all/4454.png\", name=\"frame\" ></p>\n");
    fprintf(page, "\t\t<div id=\"my-slider\" class=\"slider\">\n");
    fprintf(page, "\t\t<div class=\"handle\"> << - >> </div>\n");
    fprintf(page, "\t</div>\n");
    fprintf(page, "</body>\n");
    fprintf(page, "</html>");
    fclose(page);
}

int main(int argc, char * argv[])
{
    if (argc != 4){
        cout << "Arguments <yml-file> <bag-file> <site-destination>" << endl;
        return -1;
    }
    YAML::Node model = YAML::LoadFile(argv[1]);
    map<int, map <int, int> > histograms;
    int K = model["K"].as<int>();
    cout << "Computing histograms..." << endl;
    if (model["observations"]){
        int i = 0;
        while(model["observations"][i]){
            int  seq = model["observations"][i]["seq"].as<int>();
            vector <int> topics = model["observations"][i]["topics"].as<vector<int> >();
            histograms[seq] = topicsHistogram(topics);
            i++;
        }
    }

    //printHistograms(histograms);
    cout << "Computing maximum likelihood images..." << endl;
    vector < map<int, int> > mostLikely;
    for (unsigned int k = 0; k < K; k++){
        mostLikely.push_back(getMostLikelySeqs(k, 5, histograms));
    }

    cout << "Calculating overall histogram..." << endl;
    map<int, int> overallHist = calcOverallHist(histograms);

    cout << "Calculating normalized histograms..." << endl;
    map<int, map<int, float> > normal_hists = normalizeHistograms(histograms);

    char dataroot_name[60];
    char dir_name1[70];
    char dir_name2[70];
    sprintf(dataroot_name, "%s/site/data/", argv[3]);
    sprintf(dir_name1, "%sbest/", dataroot_name);
    sprintf(dir_name2, "%sall/", dataroot_name);
    boost::filesystem::create_directories(dir_name1);
    boost::filesystem::create_directories(dir_name2);

    cout << "Saving maximum likelihood images in " << dir_name1 << endl;
    saveImages(argv[2], "/camera_front_center/image", mostLikely, dir_name1);

    cout << "Saving all images in " << dir_name2 << endl;
    int minSeq = 100000;
    int maxSeq = 0;
    saveAllImages(argv[2], "/camera_front_center/image", dir_name2, &minSeq, &maxSeq);

    char htmlroot_name[40];
    sprintf(htmlroot_name, "%s/site/", argv[3]);
    boost::filesystem::create_directories(htmlroot_name);
    cout << "Writing all topics page..." << endl;
    writeAllTopicsPage(K, mostLikely, overallHist, htmlroot_name);

    cout << "Writing all images page..." << endl;
    writeAllImagesPage(htmlroot_name, normal_hists, minSeq, maxSeq);

    char topicsroot_name[40];
    sprintf(topicsroot_name, "%stopics/", htmlroot_name);
    boost::filesystem::create_directories(topicsroot_name);
    cout << "Writing topic pages in " << topicsroot_name << endl;
    for (unsigned int k = 0; k < K; k++){
        writeTopicPage(k, mostLikely[k], overallHist, normal_hists, topicsroot_name);
    }
    cout << "Copying javascript & css resources" << endl;
    char resourceroot_name[40];
    sprintf(resourceroot_name, "%sresource", htmlroot_name);
    if (copyDir(boost::filesystem::path("./resource"), boost::filesystem::path(resourceroot_name)) == 0){
        cout << "Error: could not copy resource directory" << endl;
        return -1;
    }

    return 0;
}
