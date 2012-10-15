#include "rost.hpp"
#include "program_options.hpp"
#include <iostream>


int main(int argc, char*argv[]){

  po::options_description options("Topic modeling of data with 1 dimensional structure.");
  load_rost_base_options(options).add_options()
    ("neighborhood.size,g", po::value<int>()->default_value(1), "Depth of the neighborhood");
  
  auto args = read_command_line(argc, argv, options);
  
  ROST<int,neighbors<int>, hash<int> > rost(args["nwords"].as<int>(),
		 args["ntopics"].as<int>(),
		 args["alpha"].as<double>(),
		 args["beta"].as<double>(),
		 neighbors<int>(args["neighborhood.size"].as<int>()),
		 hash<int>());
  


  
  word_reader in(args["in.words"].as<string>(), args["in.cellsize"].as<int>()); 


  if(!args.count("online")){
    cerr<<"Processing words in batch mode."<<endl;        
    int t=0;
    auto words = in.get();
    while(!words.empty()){
      rost.add_observation(t++,words);
      words = in.get();
    }
    cerr<<"Read "<<rost.cells.size()<<" cells"<<endl;
    for(int i=0; i<args["iter"].as<int>(); ++i){
      cerr<<"Iter: "<<i<<"    \r";
      parallel_refine(&rost, args["threads"].as<int>());
    }
    
    cerr<<"Writing output:"<<endl;
    ofstream out_topics(args["out.topics"].as<string>());
    for(int d = 0; d< rost.C; ++d){
      vector<int> topics = rost.get_topics_for_pose(d);
      copy(topics.begin(), topics.end(), ostream_iterator<int>(out_topics," "));
      //copy(rost.cells[d]->Z.begin(), rost.cells[d]->Z.end(), ostream_iterator<int>(out_topics," "));
      out_topics<<endl;
    }
  }
  else{
    cerr<<"Processing words online."<<endl;
    atomic<bool> stop;
    stop.store(false);
    auto workers =  parallel_refine_online(&rost, args["tau"].as<double>(), args["threads"].as<int>(), &stop);
    ofstream out_topics(args["out.topics"].as<string>());
    int t=0;

    auto words = in.get();
    cerr<<"adasdasd"<<endl;
    while(!words.empty()){
      this_thread::sleep_for(chrono::milliseconds(args["online.mint"].as<int>()));
      //      cerr<<"Read cell: "<<t<<endl;
      if(t>0){
	vector<int> topics = rost.get_topics_for_pose(t-1);
	copy(topics.begin(), topics.end(), ostream_iterator<int>(out_topics," "));
	out_topics<<endl;
      }
      rost.add_observation(t++,words);
      words = in.get();      
    }
    vector<int> topics = rost.get_topics_for_pose(t-1);
    copy(topics.begin(), topics.end(), ostream_iterator<int>(out_topics," "));
    //copy(rost.cells[t-1]->Z.begin(), rost.cells[t-1]->Z.end(), ostream_iterator<int>(out_topics," "));
    out_topics<<endl;
   
    cerr<<"Reached EOF. Terminating."<<endl;
    stop.store(true);
    for(auto t:workers){
      t->join();
    }
  }
  return 0;
}
