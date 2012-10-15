#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::options_description& load_rost_base_options( po::options_description & desc){
  desc.add_options()
    ("help", "help")    
    ("in.words,i",   po::value<string>()->default_value("/dev/stdin"),"Word frequency count file. Each line is a document/cell, with integer representation of words. ")
    ("in.cellsize",  po::value<int>()->default_value(0), "Instead of using newline to split words in cells, make each cell of this given size.")
    ("out.topics,o", po::value<string>()->default_value("/dev/stdout"),"Output topics file")    
    ("nwords,V",     po::value<int>()->default_value(1000), "Vocabulary size.")
    ("ntopics,K",    po::value<int>()->default_value(100), "Topic size.")
    ("iter,n",       po::value<int>()->default_value(1000), "Number of iterations")
    ("alpha,a",      po::value<double>()->default_value(0.1), "Controls the sparsity of theta. Lower alpha means the model will prefer to characterize documents by few topics")
    ("beta,b",       po::value<double>()->default_value(0.1), "Controls the sparsity of phi. Lower beta means the model will prefer to characterize topics by few words.")
    ("online,l",     "Do online learning; i.e., output topic labels after reading each document/cell.")
    ("online.mint",  po::value<int>()->default_value(0),     "Minimum time in ms to spend between new observations")
    ("tau",          po::value<double>()->default_value(10), "High the tau, more biased the refinement is towards the present. tau=1 implies no bias. tau = 0 imples only the oldest(first) cell is imporant.")
    ("threads", po::value<int>()->default_value(4),"Number of threads to use.")
    ;
  return desc;
}


po::variables_map read_command_line(int argc, char* argv[], po::options_description& options){

  po::variables_map args;
  po::store(po::command_line_parser(argc, argv)
	    .style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing)
	    .options(options)
	    .run(), 
	    args);
  po::notify(args);

  if(args.count("help")){
    cerr<<options<<endl;
    exit(0);
  }
  return args;
}
