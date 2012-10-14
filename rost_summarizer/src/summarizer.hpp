#ifndef SUMAMRIZER_203920
#define SUMAMRIZER_203920
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <set>
#include <limits>
#include "boost/tuple/tuple.hpp"
#include <cmath>
#include <numeric>
using namespace std;
using namespace boost;


template <typename T>
class KLDistance{
public:
  T operator()(const vector<T>& p, const vector<T>& q) const{
    //    if(p.size() != q.size()){
    //      cerr<<"p.size()="<<p.size()<<"  q.size()="<<q.size()<<endl;
    //      assert(p.size() == q.size());
    //    }
    //    size_t len = std::min(p.size(), q.size());
    //T p_total = std::accumulate(p.begin(), p.begin()+len,0);
    //    T q_total = std::accumulate(q.begin(), q.begin()+len,0);
    T pv, qv;
    T d(0);
    typename vector<T>::const_iterator pi=p.begin(), p_end=p.end(), qi=q.begin(), q_end=q.end();
    //    for(size_t i=0;i<len; ++i){
      //      d+= ((p[i]==0 && q[i]==0) ? 0: (p[i]/p_total)*log((p[i]/p_total)/(q[i]/q_total)));
      
    //      d+= ((p[i]==0 && q[i]==0) ? 0: (p[i]*log(p[i]/q[i])));
    //    }

    for(;pi!=p_end && qi!=q_end; ++pi, ++qi){
      pv = *pi; qv=*qi;
      d+= ((pv==0 && qv==0) ? 0: (pv * log(pv/qv )));
    }
    for(;pi!=p_end; ++pi){
      pv = *pi; qv=0;
      d+= ((pv==0 && qv==0) ? 0: (pv * log(pv/qv )));
    }
    for(;qi!=q_end; ++qi){
      pv = 0; qv=*qi;
      d+= ((pv==0 && qv==0) ? 0: (pv * log(pv/qv )));
    }

    if(d<0){
      cerr<<"WARNING: dKL="<<d<<endl;
      //assert(false);
      d=0;
    }
    /*    if(d==std::numeric_limits<T>::infinity()){
      cerr<<"d=inf"<<endl;
      copy(p.begin(),p.end(),ostream_iterator<T>(cerr," ")); cerr<<endl;
      copy(q.begin(),q.end(),ostream_iterator<T>(cerr," ")); cerr<<endl;
      for(size_t i=0;i<p.size(); ++i){
	cerr<<p[i]<<"*log("<<p[i]<<"/"<<q[i]<<")="<<p[i]*log(p[i]/q[i])<<"  "	;
      }
      cerr<<endl;
      }*/
    return d;
  }
};

template<typename Distance=KLDistance<float> >
class Summary{
public:
  enum ThresholdingStrategy {TH_auto, TH_min, TH_2min, TH_mean, TH_median, TH_doubling};
  ThresholdingStrategy thresholding;
  size_t last_summary_size;
  typedef vector<float> obs_t;
  typedef multimap<unsigned int,obs_t > summary_t;  
  typedef multimap<unsigned int,obs_t > observation_set_t;  
  typedef summary_t::iterator iterator;
  typedef set<unsigned int> uids_t;
  summary_t summary;   //stores the current summary
  unsigned int K_max;         // K_max is the max size of the summary, if set to 0, implies summary size is allowed to increase over time
  Distance distance;  ///distance function used to compute distance between two observations
  float threshold;  //current recommended picking threshold based on the current summary
  uids_t uids; //stores the unique observation ids currently in the summary
  Summary(unsigned int K=0, const string& thresholding="auto", const Distance& distance = Distance());

  //add observations to the summary
  void add(const obs_t& topics, size_t id);
  //add a set of sub_observations, each with the same observation id
  template<typename ObsIt>
  void add(ObsIt begin, ObsIt end, size_t id){
    for(ObsIt i = begin; i!=end; ++i)
      summary.insert(make_pair(id, *i));
    
    uids.insert(id);  //save the ids.
    if(K_max>0 && uids.size() > K_max){
      trim();
    }
  }

  void remove(size_t id);
  bool has(size_t id);

  tuple<float,unsigned int> surprise(const obs_t& observation);

  template<typename ObsIt, typename OutSurpIt>
  tuple<float,unsigned int, ObsIt> surprise(ObsIt begin, ObsIt end, OutSurpIt surp_it){
    observation_set_t::iterator it, max_it=summary.end();
    ObsIt max_obsit = end;
    float s = 0, max_s=0;
    
    for(ObsIt oit = begin; oit!=end; ++oit){
      tie(s,it)=hausdorff(*oit, summary.begin(), summary.end());
      *surp_it++ = s;
      if(s>max_s){
	max_it = it;
	max_s = s;
	max_obsit = oit;
      }
    }
    return make_tuple(s, max_it->first, max_obsit);

  }

  void trim();
  size_t num_observation_sets(){
    return uids.size();
  }
  void update_threshold();

  //retunrs argmin_i dist(o, sum[i])
  tuple<float, observation_set_t::iterator> 
  hausdorff(const obs_t& o, 
	    observation_set_t::iterator  sum_begin, observation_set_t::iterator sum_end) const
  {
    summary_t::iterator min_it,it;
    min_it = it = sum_end;
    float min_d = numeric_limits<float>::max();

    for(it = sum_begin; it!=sum_end; ++it){
      float d = distance(it->second, o);
      if(d<min_d){ //min
	min_it = it;
	min_d = d;
      }
    }
    return make_tuple(min_d,min_it);
  }
  tuple<float, observation_set_t::iterator> 
  hausdorff(observation_set_t::iterator o_it, 
	    observation_set_t::iterator  sum_begin, observation_set_t::iterator sum_end) const
  {
    summary_t::iterator min_it,it;
    min_it = it = sum_end;
    float min_d = numeric_limits<float>::max();

    for(it = sum_begin; it!=sum_end; ++it){
      if(it->first==o_it->first) continue; //ignore if both have the same observation id
      float d = distance(it->second, o_it->second);
      if(d<min_d){ //min
	min_it = it;
	min_d = d;
      }
    }
    return make_tuple(min_d,min_it);
  }

  /*  tuple<float, observation_set_t::iterator> 
  hausdorff(const observation_set_t::iterator& o_it, 
	    const observation_set_t & sum) const
  {  
    return hausdorff(o_it, sum.begin(), sum.end());
  }
  */

  //returns argmax_i hausdorff(sum1[i], sum2)
  tuple<float, observation_set_t::iterator, observation_set_t::iterator> 
  hausdorff(observation_set_t::iterator  sum1_begin, observation_set_t::iterator sum1_end, 
	    observation_set_t::iterator  sum2_begin, observation_set_t::iterator sum2_end){
    observation_set_t::iterator max_it,min_it, it1, it2;
    max_it = sum1_end;
    min_it = sum2_end;
    float max_d = 0;
    float d;
    
    for(it1 = sum1_begin; it1!=sum1_end; ++it1){
      tie(d, it2) = hausdorff(it1, sum2_begin, sum2_end);
      if(d > max_d){
	max_it = it1;
	min_it = it2;
	max_d = d;
      }
    }
    return make_tuple(max_d, max_it, min_it);
  }
  /*  tuple<float, observation_set_t::iterator, observation_set_t::iterator> 
  hausdorff(const observation_set_t & sum1, 
	    const observation_set_t & sum2){  
    return hausdorff(sum1.begin(), sum1.end(), sum2.begin(), sum2.end());
    }*/

  tuple<float, observation_set_t::iterator, observation_set_t::iterator> 
  symmetric_hausdorff(observation_set_t::iterator  sum1_begin, observation_set_t::iterator sum1_end, 
		      observation_set_t::iterator  sum2_begin, observation_set_t::iterator sum2_end){
    observation_set_t::iterator left_it1, left_it2, right_it1, right_it2;
    float d_left, d_right;
    tie(d_left, left_it1, left_it2) = hausdorff(sum1_begin, sum1_end, sum2_begin, sum2_end);
    tie(d_right, right_it2, right_it1) = hausdorff(sum2_begin, sum2_end, sum1_begin, sum1_end);
    return (d_left > d_right)? tie(d_left, left_it1, left_it2) : tie(d_right, right_it1, right_it2);            
  }
  /*  tuple<float, observation_set_t::iterator, observation_set_t::iterator> symmetric_hausdorff(const observation_set_t & sum1, const observation_set_t & sum2){
    return symmetric_hausdorff(sum1.begin(), sum1.end(), sum2.begin(), sum2.end());
    }*/

};




template<typename Distance>
Summary<Distance>::Summary(unsigned int K_, const string& thresholding_, const Distance& distance_):
  K_max(K_), 
  distance(distance_),
  threshold(0)
{  
  cerr<<"Initialized summarizer with thresholding strategy: ";
  if(thresholding_ == "2min"){
    thresholding = TH_2min;
    cerr<<"2*min";
  }
  if(thresholding_ == "min"){
    thresholding = TH_min;
    cerr<<"min";
  }
  else if (thresholding_ == "doubling"){
    thresholding = TH_doubling;
    cerr<<"doubling";
  }
  else if (thresholding_ == "mean"){
    thresholding = TH_mean;
    cerr<<"mean";
  }
  else if (thresholding_ == "median"){
    thresholding = TH_median;
    cerr<<"median";
  }
  else{
    if(K_max>1){//fixed summary size
      thresholding = TH_2min;
      cerr<<"auto=>2*min";
    }
    else{
      thresholding = TH_mean;
      cerr<<"auto=>mean";
    }
  }
  cerr<<endl;
}

template<typename Distance>
void Summary<Distance>::add(const obs_t& observation, size_t id){
  summary.insert(make_pair(id, observation));
  uids.insert(id);  //save the ids.
  if(K_max>0 && uids.size() > K_max){
    trim();
  }
}

template<typename Distance>
void Summary<Distance>::remove(size_t id){
  summary.erase(id);
  uids.erase(id);  //save the ids.
}

template<typename Distance>
bool Summary<Distance>::has(size_t id){
  if(uids.find(id)!=uids.end()){
    //assert(summary.find(id)!=summary.end());
    return true;    
  }
  return false;
}

template<typename Distance>
tuple<float,unsigned int> Summary<Distance>::surprise(const obs_t& observation){
  summary_t::iterator it;
  float s;
  tie(s,it)=hausdorff(observation, summary.begin(), summary.end());
  if(it!=summary.end()){
    return make_tuple(s, it->first);
  }
  else
    return make_tuple(0.0f, -1);
}


template<typename Distance>
void Summary<Distance>::trim(){
  if(summary.empty())
    return;
  assert(uids.size()>0);

  unsigned int new_k = uids.size()-1;
  //keep the last selected image;
  summary_t::iterator i = summary.end(), i2;
  float d;
  --i;

  summary_t newsum; set<unsigned int> new_uids;
  newsum.insert(summary.lower_bound(i->first), summary.upper_bound(i->first)); //insert all elemets with key i->first
  new_uids.insert(i->first);
  //  newsum[i->first]=summary[i->first];
  summary.erase(i->first);
  new_k--;
  while(new_k>0){
    assert(!summary.empty());
    tie(d,i,i2) = hausdorff(summary.begin(), summary.end(), newsum.begin(), newsum.end());
    newsum.insert(summary.lower_bound(i->first), summary.upper_bound(i->first)); //insert all subsampels with key i-first
    new_uids.insert(i->first);
    //    newsum[i->first]=i->second;
    summary.erase(i->first);
    new_k--;
  }
  summary.swap(newsum);
  uids.swap(new_uids);
}



template<typename Distance>
void Summary<Distance>::update_threshold(){
  if(uids.size()<2){
    threshold=0;
    return;
  }
  vector<float> dists;
  float total_d=0;
  float min_d=std::numeric_limits<float>::max();
  //summary_surprise.clear();
  vector<float> surprise_scores;

  //  tie(min_d, tuples::ignore, tuples::ignore) = hausdorff(summary.begin(), summary.end(), summary.begin(), summary.end())
  for(uids_t::iterator it = uids.begin(); it!=uids.end(); ++it){
    float d;
    summary_t::iterator it1, it2; 
    assert(summary.lower_bound(*it)!=summary.end());
    tie(d,it1,it2)=hausdorff(summary.lower_bound(*it), summary.upper_bound(*it), summary.begin(), summary.end());
    //cerr<<"surprise score: "<<*it<<" --> "<<d<<endl;
    surprise_scores.push_back(d);
    if(d<min_d)
      min_d = d;
    total_d+=d;
  }
 

  /*  for(summary_t::iterator it = summary.begin(); it!=summary.end(); ++it){
    float d;
    tie(d,tuples::ignore) = hausdorff(it, summary.begin(), summary.end());
    surprise_scores.push_back(d);
    if(d<min_d)
      min_d = d;
    total_d+=d;
    }*/

  //cerr<<"Total_d: "<<total_d<<endl;
  if(uids.size()<K_max && K_max>0){
    threshold=0;
  }
  else{
    switch(thresholding){      
    case TH_min: threshold= min_d; break;
    case TH_2min: threshold= min_d*2; break;
    case TH_doubling: 
      if(uids.size()==2 && threshold==0) 
	threshold = 2*min_d; 
      else if(last_summary_size < uids.size())
	threshold *= 2.0;
      break;
    case TH_median:
      sort(surprise_scores.begin(), surprise_scores.end());
      threshold = surprise_scores[surprise_scores.size()/2];
      break;
    case TH_mean: 
    default:
      threshold= total_d/uids.size();
    };
  }

  last_summary_size = uids.size();
}

#endif
