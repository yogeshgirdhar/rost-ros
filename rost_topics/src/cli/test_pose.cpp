#include "pose.hpp"
#include <iostream>
#include <chrono>
#include <set>
#include <thread>
#include <mutex>
using namespace std;

typedef array<int,3> Pose;

/*
void f(){
}
struct Refinery;

void dowork(ROST<Pose>* rost, Refinery* ref){
  while(true){
    size_t cid = ref->get_cell();
    rost->refine(rost->cells[cid]);
    ref->put_cell(cid);
  }
}

struct Refinery{
  ROST<Pose>& rost;
  vector<thread> threads;
  mutex busy_mutex;
  set<size_t> busy;

  size_t get_cell(){
    lock_guard<mutex> lock(busy_mutex);
    size_t cid;
    do{
      cid= rand()%(rost.cells.size());
    }while(busy.find(cid)!=busy.end());
    busy.insert(cid);
    return cid;
  }

  void put_cell(size_t cid){
    lock_guard<mutex> lock(busy_mutex);
    assert(busy.erase(cid) == 1);
  }

  Refinery(ROST<Pose>& r, int num_threads): rost(r){
    for(int i=0; i<num_threads; ++i){
      auto w = std::bind(dowork, &rost, this);
      threads.emplace_back(w);
    }
  }

  void join(){
    for(auto& t: threads){
      t.join();
    }
  }
};
*/

mutex busy_mutex;
set<size_t> busy;
void dowork(ROST<Pose>* rost){
  while(true){
    size_t cid;
    busy_mutex.lock();
    do{
      cid= rand()%(rost->cells.size());
    }while(busy.find(cid)!=busy.end());
    busy.insert(cid);
    busy_mutex.unlock();

    //    cerr<<"Refininig: "<<cid<<" in thread: "<<endl;
    rost->refine(rost->cells[cid]);
    //cerr<<"END Refininig: "<<cid<<" in thread: "<<endl;

    busy_mutex.lock();
    assert(busy.erase(cid)==1);
    busy_mutex.unlock();
  }
}


void dowork_parallel_refine(ROST<Pose>* rost,shared_ptr<vector<size_t>> todo, shared_ptr<mutex> m){
  while(!todo->empty()){
    m->lock();
    size_t cid = todo->back(); todo->pop_back();
    m->unlock();
    //cerr<<"Refininig: "<<cid<<" in thread: "<<this_thread::get_id()<<endl;
    rost->refine(rost->cells[cid]);
    //cerr<<"END Refininig: "<<cid<<" in thread: "<<endl;
  }
  
}

void parallel_refine(ROST<Pose>* rost, int nt){
  auto todo = make_shared<vector<size_t>>(rost->cells.size());
  random_shuffle(todo->begin(), todo->end());
  for(size_t i=0;i<todo->size(); ++i){
    (*todo)[i]=i;
  }
  auto m = make_shared<mutex>();
  vector<shared_ptr<thread>> threads;
  for(int i=0;i<nt;++i){
    //threads.emplace_back(dowork_parallel_refine,rost,todo ,m);
    threads.push_back(make_shared<thread>(dowork_parallel_refine,rost,todo ,m));
  }
  
  for(auto&t: threads){
    t->join();
  }
}


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
