#include "rost.hpp"
#include <iostream>
#include <random>
using namespace std;


int main(int argc, char*argv[]){
  if(argc!=5){
    cerr<<"Generate random words from topics"<<endl
	<<"  generate_random_words <topics.txt> <K> <V> <alpha>"<<endl;
    return 0;
  }
  int K=2, V=10, seed;
  double alpha;

  word_reader reader(argv[1]);
  K = atoi(argv[2]);
  V = atoi(argv[3]);
  alpha = atof(argv[4]);

  //  seed = atoi(argv[4]);
  random_device rd;
  mt19937 engine(rd());
  vector<discrete_distribution<>> topic_word_dist;
  
  vector<int> vocabulary(V);
  for(int i=0;i<V; ++i){
    vocabulary[i]=i;
  }
  random_shuffle(vocabulary.begin(), vocabulary.end());

  int words_per_topic = V/K;
  for(int k=0;k<K;++k){
    vector<double> dist(V,alpha);
    for(int i= k*words_per_topic; i< (k+1)*words_per_topic; ++i){
      dist[i]+=1.0;
    }
    topic_word_dist.emplace_back(dist.begin(), dist.end());
  }
  vector<int> topics = reader.get();
  while(!topics.empty()){
    for(auto k: topics){
      cout<<vocabulary[topic_word_dist[k](engine)]<<" ";
    }
    cout<<endl;
    topics = reader.get();
  }

  return 0;
}
