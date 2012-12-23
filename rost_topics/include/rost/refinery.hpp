#ifndef ROST_TOPICS_REFINERY_HPP
#define ROST_TOPICS_REFINERY_HPP

#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include <mutex>
#include <memory>
#include <thread>
#include <iterator>
#include <atomic>
using namespace std;
/*
  Following functions are used to coordinate parallel topic refinement
 */


template<typename R>
void dowork_parallel_refine(R* rost,shared_ptr<vector<size_t>> todo, shared_ptr<mutex> m, int thread_id){
  while(true){
    m->lock();
    if(todo->empty()){m->unlock(); break;}
    size_t cid = todo->back(); todo->pop_back();
    //    cerr<<"T:"<<thread_id<<" >"<<cid<<endl;
    m->unlock();
    rost->refine(*(rost->get_cell(cid)));
  }  
}

template<typename R>
void parallel_refine(R* rost, int nt){
  auto todo = make_shared<vector<size_t>>(rost->C);
  for(size_t i=0;i<todo->size(); ++i){
    (*todo)[i]=i;
  }
  random_shuffle(todo->begin(), todo->end());
  auto m = make_shared<mutex>();
  
  vector<shared_ptr<thread>> threads;
  for(int i=0;i<nt;++i){
    //threads.emplace_back(dowork_parallel_refine,rost,todo ,m);
    threads.push_back(make_shared<thread>(dowork_parallel_refine<R>,rost,todo ,m, i));
  }
  
  for(auto&t: threads){
    t->join();
  }
}

// does #iter number of cell refinements
// cell_ids to be refined are distributed by Beta(tau,1)
// tau > 1 => higher cell_ids are refined more often
// nt = number of threads to use
template<typename R>
void parallel_refine_tau(R* rost, int nt, double tau, size_t iter){
  auto todo = make_shared<vector<size_t>>(iter);
  gamma_distribution<double> gamma1(tau,1.0);
  gamma_distribution<double> gamma2(1.0,1.0);
  for(size_t i=0;i<iter; ++i){
    double r_gamma1 = gamma1(rost->engine), r_gamma2 = gamma2(rost->engine);
    double r_beta = r_gamma1/(r_gamma1+r_gamma2);
    size_t cid = floor(r_beta * static_cast<double>(rost->C));
    cid = min<size_t>(cid, rost->C-1);
    (*todo)[i]=cid;
  }
  auto m = make_shared<mutex>();  
  vector<shared_ptr<thread>> threads;
  for(int i=0;i<nt;++i){
    //threads.emplace_back(dowork_parallel_refine,rost,todo ,m);
    threads.push_back(make_shared<thread>(dowork_parallel_refine<R>,rost,todo ,m, i));
  }
  
  for(auto&t: threads){
    t->join();
  }
}

/// refine the last <current_size> cells with uniform distribution  with with probability <p_refine_current>
/// refine all cells with distribution Beta(tau,1) with probability 1-<p_refine_current>
template<typename R, typename Stop>
void dowork_parallel_refine_online2(R* rost, double tau, double p_refine_current, size_t current_size, int thread_id, Stop stop){
  gamma_distribution<double> gamma1(tau,1.0);
  gamma_distribution<double> gamma2(1.0,1.0);
 
  cerr<<"Initializing refine thread "<<thread_id<<" with tau="<<tau<<" p_refine_current="<<p_refine_current<<"  current_size="<<current_size<<endl;
  while(! stop->load() ){    
    if(rost->C > 0){
      size_t cid;
      double r_refine_current = generate_canonical<double, 20>(rost->engine);
      if(r_refine_current < p_refine_current){
	//uniformly pick a cell from cells in the current observation.
	double p_refine = generate_canonical<double, 20>(rost->engine);
	cid = rost->C - min<size_t>(current_size,rost->C) + floor(p_refine * current_size);
	//cerr<<"refine local: "<<cid<<endl;
      }
      else{
	double r_gamma1 = gamma1(rost->engine), r_gamma2 = gamma2(rost->engine);
	double r_beta = r_gamma1/(r_gamma1+r_gamma2);
	//pick a cell from beta(tau,1)*C
	cid = floor(r_beta * static_cast<double>(rost->C));
	//cerr<<"refine global: "<<cid<<endl;
      }

      if(cid >= rost->C) 
	cid = rost->C-1;
      //cerr<<"final: "<<cid<<endl;
      if(rost->get_cell(cid)->cell_mutex.try_lock()){
	//	cerr<<"T:"<<thread_id<<" >"<<cid<<endl;
	rost->refine(*rost->get_cell(cid));
	rost->get_cell(cid)->cell_mutex.unlock();
      }
    }
  }
}


template<typename R, typename Stop>
void dowork_parallel_refine_online(R* rost, double tau, int thread_id, Stop stop){
  gamma_distribution<double> gamma1(tau,1.0);
  gamma_distribution<double> gamma2(1.0,1.0);
  size_t now_size = 200;
 
  while(! stop->load() ){
    double r_gamma1 = gamma1(rost->engine), r_gamma2 = gamma2(rost->engine);
    double r_beta = r_gamma1/(r_gamma1+r_gamma2);
    double p_refine_current = generate_canonical<double, 20>(rost->engine);
    if(rost->C > 0){
      size_t cid;
      if(p_refine_current < 0.9 || rost->C < now_size || tau==1.0){
	cid = floor(r_beta * static_cast<double>(rost->C));
	//cerr<<"global: "<<cid<<endl;
      }
      else{
	cid = max<int>(0,rost->C - now_size + floor(r_beta * now_size));
	//cerr<<"local: "<<cid<<"/"<<rost->C<<endl;
      }
      if(cid >= rost->C) 
	cid = rost->C-1;

      assert(cid < rost->C);
      if(rost->get_cell(cid)->cell_mutex.try_lock()){
	//	cerr<<"T:"<<thread_id<<" >"<<cid<<endl;
	rost->refine(*rost->get_cell(cid));
	rost->get_cell(cid)->cell_mutex.unlock();
      }
    }
  }
}

template<typename R, typename Stop>
vector<shared_ptr<thread>> parallel_refine_online(R* rost, double tau, int nt, Stop stop){  
  cerr<<"Spawning "<<nt<<" worker threads for refining topics"<<endl;
  vector<shared_ptr<thread>> threads;
  for(int i=0;i<nt;++i){
    threads.push_back(make_shared<thread>(dowork_parallel_refine_online<R,Stop>,rost,tau, i, stop));
  }
  return threads;
}

template<typename R, typename Stop>
vector<shared_ptr<thread>> parallel_refine_online2(R* rost, double tau, double p_refine_current, size_t current_size, int nt, Stop stop){  
  cerr<<"Spawning "<<nt<<" worker threads for refining topics"<<endl;
  vector<shared_ptr<thread>> threads;
  for(int i=0;i<nt;++i){
    threads.push_back(make_shared<thread>(dowork_parallel_refine_online2<R,Stop>,rost,tau, p_refine_current, current_size, i, stop));
  }
  return threads;
}


#endif
