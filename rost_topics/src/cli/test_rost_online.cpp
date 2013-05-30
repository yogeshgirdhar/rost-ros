#include "rost.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <thread>
#include <mutex>
using namespace std;

typedef array<int,3> Pose;

int main(){

  Pose a = {0,0,0};
  int V=10000,K=100; 
  ROST<Pose> rost(V,K,0.1,0.1, neighbors<Pose>(2));

  uniform_int_distribution<int> dim_distr(0,tuple_size<Pose>::value);
  uniform_int_distribution<int> step_distr(-1,1);
  uniform_int_distribution<int> word_distr(0,V-1);
  uniform_int_distribution<int> cell_size_distr(1,5);


  atomic<bool> stop;
  stop.store(false); 
  auto workers =  parallel_refine_online(&rost, 2, 4, &stop);

  while(true){
    vector<int> words(cell_size_distr(rost.engine));
    a[dim_distr(rost.engine)]+=step_distr(rost.engine);    
    generate(words.begin(), words.end(), bind(word_distr,rost.engine));
    cerr<<"+("<<words.size()<<")";
    rost.add_observation(a,words);
    std::this_thread::sleep_for(  std::chrono::milliseconds( 100 ));
  }


  return 0;
}
