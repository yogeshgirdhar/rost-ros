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
  cout<<"Origin: "<<a<<endl;

  cout<<"Neighbors: \n";
  neighbors<Pose>g2(2);
  for(auto g : g2(a)){
    cout<<g<<endl;
  }

  cout<<"Sizeof(size_t): "<<sizeof(size_t)<<endl;
  hash_container<Pose> hash_pose;
  cout<<"Hash for origin: "<<hash_pose(a)<<endl;   
  cout<<"Hash for neighbors: "<<endl;
  for(auto g : g2(a)){
    cout<<hash_pose(g)<<endl;
  } 
  


  ROST<Pose> rost(10000,100,0.1,0.1, neighbors<Pose>(2));

  uniform_int_distribution<int> dim_distr(0,tuple_size<Pose>::value);
  uniform_int_distribution<int> step_distr(-1,1);
  uniform_int_distribution<int> word_distr(0,9999);
  vector<int> words(1000);
  for(int i=0;i<1000;++i){
    a[dim_distr(rost.engine)]+=step_distr(rost.engine);    
    //    cout<<a<<endl;
    generate(words.begin(), words.end(), bind(word_distr,rost.engine));
    rost.add_observation(a, words);
  }

  chrono::system_clock clock;
  chrono::time_point<chrono::system_clock> start_time, end_time;
  //  Refinery refinery(rost, 4);
  for(int i=0;i<100;++i){
    start_time = clock.now();
    parallel_refine(&rost,4);
    end_time = clock.now();
    cout<<"Duration: "<<chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count()<<endl;
  }
  return 0;
}
