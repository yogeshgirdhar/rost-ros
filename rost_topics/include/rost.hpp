#include <array>
#include <vector>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <mutex>
#include <memory>
#include <thread>
#include <iterator>
#include <atomic>

#include "rost/refinery.hpp"
#include "rost/word_reader.hpp"
using namespace std;


typedef array<int,6> Pose6i;
typedef array<int,3> Pose3i;
typedef array<int,1> Posei;


ostream& operator<<(ostream& out, const array<int,3>& v){
  for(auto a : v){
    out<<a<<" ";
  }  
  return out;
}
ostream& operator<<(ostream& out, const array<int,6>& v){
  for(auto a : v){
    out<<a<<" ";
  }  
  return out;
}


/*ostream& operator<<(ostream& out, const Pose6i& v){
  for(auto a : v){
    out<<a<<" ";
  }  
  return out;
  }*/


//maps the v to [-N,N)
template<typename T>
T standardize_angle( T v, T N=180){  
  T r = v%(2*N);
  r = r >= N ? r-2*N: r; 
  r = r < -N ? r+2*N: r;
  return r;
}

template<typename Pose>
int pose_distance(const Pose& p1, const Pose& p2){
  int d=0;
  for(size_t i=0;i<p1.size(); ++i){
    d+= std::abs(p1[i] - p2[i]);
  }
  return d;
}

template<>
int pose_distance<int>(const int& p1, const int& p2){
  return std::abs(p1-p2);
}

//neighbors functor returns the neighbors of the given pose 
//template<typename T>
//struct neighbors{
//  vector<T> operator()(const T &o) const;
//};

//neighbors specialization for a pose of array type
template<typename Array>
struct neighbors {
  //  const int depth;
  Array depth;
  size_t neighborhood_size;
  neighbors( int depth_){
    depth.fill(depth_);
    neighborhood_size=depth.size()*2*depth[0];
  }
  neighbors( const Array& depth_):depth(depth_){
    neighborhood_size=0;
    for(auto d:  depth){
      neighborhood_size += 2*d;
    }
  }
  vector<Array> operator()(const Array& o) const{

    vector<Array> neighbor_list(neighborhood_size, o);
    
    auto outit = neighbor_list.begin();
    for(size_t i=0; i<o.size(); ++i){
      for(int d = 0; d<depth[i]; ++d){
	(*outit++)[i] += d+1;
	(*outit++)[i] -= d+1;
      }
    }
    return neighbor_list;
  }
};

template<>
struct neighbors<int> {
  int depth;
  neighbors(int depth_):depth(depth_){}
  vector<int> operator()(int o) const{
    vector<int> neighbor_list(2*depth,o);
    auto i = neighbor_list.begin();
    for(int d=1;d<=depth; ++d){
      (*i++)+=d;
      (*i++)-=d;
    }
    return neighbor_list;
  }
};


template<typename T>
struct hash_container{
  hash<typename T::value_type> hash1;
  size_t operator()(const T& pose) const {
    size_t h(0);
    for(size_t i=0;i<pose.size(); ++i){
      if(sizeof(size_t)==4){
      	h = (h<<7) | (h >> 25);
      }
      else{
	h = (h<<11) | (h >> 53);
      }
      h = hash1(pose[i]) ^ h;
    }
    return h;
  }
};


/*template<>
struct hash_container<int>{
  hash<int> hash1;
  size_t operator()(int pose) const {
  return hash1(pose);
  }
};
*/

struct Cell{
  size_t id;
  vector<size_t> neighbors;
  vector<int> W; //word labels
  vector<int> Z; //topic labels
  //vector<int> nW; //distribution/count of W
  vector<int> nZ; //distribution/count of Z
  mutex cell_mutex;
  vector<mutex> Z_mutex;
  double perplexity;
  Cell(size_t id_, size_t vocabulary_size, size_t topics_size):
    id(id_),
    //nW(vocabulary_size, 0),
    nZ(topics_size, 0),
    Z_mutex(topics_size)
  {
  }
  vector<int> get_topics(){
    lock_guard<mutex> lock(cell_mutex);
    return Z;
  }
  pair<int, int> get_wz(int i){
    cell_mutex.lock();
    auto r=make_pair(W[i],Z[i]);
    cell_mutex.unlock();
    return r;
  }
  void relabel(size_t i, int z_old, int z_new){
    if(z_old == z_new)
      return;
    lock(Z_mutex[z_old], Z_mutex[z_new]);
    Z[i]=z_new;
    nZ[z_old]--;
    nZ[z_new]++;    
    Z_mutex[z_old].unlock();  Z_mutex[z_new].unlock();
  }
  void shrink_to_fit(){
    neighbors.shrink_to_fit();
    W.shrink_to_fit();
    Z.shrink_to_fit();
    //nW.shrink_to_fit();
    nZ.shrink_to_fit();
    Z_mutex.shrink_to_fit();
  }
};

template<typename T>
struct scaled_plus{
  T scale;
  scaled_plus(const T& scale_):scale(scale_){}
  T operator()(const T& v1, const T& v2) const{
    return v1 + v2/scale;
  }
};

template<typename PoseT, 
	 typename PoseNeighborsT=neighbors<PoseT>, 
	 typename PoseHashT=hash_container<PoseT> >
struct ROST{
  PoseNeighborsT neighbors; //function to compute neighbors
  PoseHashT pose_hash;
  unordered_map<PoseT, size_t , PoseHashT> cell_lookup;
  vector<shared_ptr<Cell>> cells;
  vector<PoseT> cell_pose;
  mutex cells_mutex;     //lock for cells, since cells can grow in size
  size_t V, K, C;        //vocab size, topic size, #cells
  double alpha, beta;
  mt19937 engine;
  //ranlux24_base engine;
  //minstd_rand0 engine;

  uniform_int_distribution<int> uniform_K_distr;
  vector<vector<int>> nZW; //nZW[z][w] = freq of words of type w in topic z
  vector<int> weight_Z;
  vector<mutex> global_Z_mutex;
  atomic<size_t> refine_count; //number of cells refined till now;

  ROST(size_t V_, size_t K_, double alpha_, double beta_, const PoseNeighborsT& neighbors_ = PoseNeighborsT(), const PoseHashT& pose_hash_ = PoseHashT()):
    neighbors(neighbors_),
    pose_hash(pose_hash_),
    cell_lookup(1000000, pose_hash),
    V(V_), K(K_), C(0),
    alpha(alpha_), beta(beta_),
    uniform_K_distr(0,K-1),
    nZW(K,vector<int>(V,0)),
    weight_Z(K,0),
    global_Z_mutex(K),
    refine_count(0)
  {
  }

  decltype(weight_Z) get_topic_weights(){
    return weight_Z;
  }

  //returns the KxV topic-word distribution matrix
  decltype(nZW) get_topic_model(){
    return nZW;
  }

  //compute maximum likelihood estimate for topics in the cell for the given pose
  vector<int> get_topics_for_pose(const PoseT& pose){
    //lock_guard<mutex> lock(cells_mutex);
    auto cell_it = cell_lookup.find(pose);
    if(cell_it != cell_lookup.end()){ 
      auto c = get_cell(cell_it->second);
      lock_guard<mutex> lock(c->cell_mutex);
      return estimate(*c);
    }
    else
      return vector<int>();
  }

  //compute maximum likelihood estimate for topics in the cell for the given pose
  tuple<vector<int>,double> get_topics_and_ppx_for_pose(const PoseT& pose){
    //lock_guard<mutex> lock(cells_mutex);
    vector<int> topics;
    double ppx =0;
    auto cell_it = cell_lookup.find(pose);
    if(cell_it != cell_lookup.end()){ 
      auto c = get_cell(cell_it->second);
      lock_guard<mutex> lock(c->cell_mutex);
      topics = estimate(*c,true);
      ppx = c->perplexity;
    }
    return make_tuple(topics,ppx);
  }

  shared_ptr<Cell> get_cell(size_t cid){
    lock_guard<mutex> lock(cells_mutex);
    return cells[cid];
  }

  size_t get_refine_count(){
    return refine_count.load();
  }
  size_t num_cells(){
    return C;
  }

  void add_count(int w, int z){
    lock_guard<mutex> lock(global_Z_mutex[z]);
    nZW[z][w]++;
    weight_Z[z]++;
  }

  void relabel(int w, int z_old, int z_new){
    //    cerr<<"lock: "<<z_old<<"  "<<z_new<<endl;
    if(z_old == z_new) return;
    //lock(global_Z_mutex[z_old<z_new?z_old:z_new ], global_Z_mutex[z_old<z_new?z_new:z_old]);
    //if(z_old >= static_cast<int>(global_Z_mutex.size())){
    //      cerr<<"z_old="<<z_old<<endl;
    // }
    //if(z_new >= static_cast<int>(global_Z_mutex.size())){
    //  cerr<<"z_new="<<z_new<<endl;
    // }
    assert(z_old < static_cast<int>(global_Z_mutex.size()));
    assert(z_new < static_cast<int>(global_Z_mutex.size()));
    if(z_old<z_new){      
      global_Z_mutex[z_old].lock();
      global_Z_mutex[z_new].lock();
    }
    else{
      global_Z_mutex[z_new].lock();
      global_Z_mutex[z_old].lock();
    }
    //    cerr<<"L:"<<z_old<<","<<z_new<<endl;
    nZW[z_old][w]--;
    weight_Z[z_old]--;
    nZW[z_new][w]++;
    weight_Z[z_new]++;
    global_Z_mutex[z_old].unlock();
    global_Z_mutex[z_new].unlock();
    //cerr<<"U:"<<z_old<<","<<z_new<<endl;
  }

  void shuffle_topics(){
    for(auto &c : cells){
      lock_guard<mutex> lock(c->cell_mutex);
      for(size_t i=0;i<c->Z.size(); ++i){
	int z_old = c->Z[i];
	int w = c->W[i];
	int z_new=uniform_K_distr(engine);
	c->nZ[z_old]--;
	c->nZ[z_new]++;
	nZW[z_old][w]--;
	nZW[z_new][w]++;
	weight_Z[z_old]--;
	weight_Z[z_new]++;
	c->Z[i]=z_new;
      }
    }
  }

  template<typename WordContainer>
  void add_observation(const PoseT& pose, const WordContainer& words){
    auto cell_it = cell_lookup.find(pose);
    bool newcell = false;
    shared_ptr<Cell> c;
    if(cell_it == cell_lookup.end()){
      c = make_shared<Cell>(C,V,K);
      cells_mutex.lock();
      cells.push_back(c);
      cell_pose.push_back(pose);
      cells_mutex.unlock();

      c->cell_mutex.lock();
      //add neighbors to the cell
      for(auto& g : neighbors(pose)){
	auto g_it = cell_lookup.find(g);
	if(g_it != cell_lookup.end()){
	  auto gc = get_cell(g_it->second);
	  //  cerr<<gc->id<<" ";
	  gc->neighbors.push_back(c->id);
	  c->neighbors.push_back(gc->id);
	}
      }
      //      cerr<<endl;
      cell_lookup[pose]=c->id;
      newcell=true;
    }
    else{
      c = get_cell(cell_it -> second);
      c->cell_mutex.lock();
    }


    //do the insertion
    //c->W.resize(c->W.size()+words.size());
    //c->Z.resize(c->Z.size()+words.size());
    for(auto w : words){
      c->W.push_back(w);
      //generate random topic label
      int z = uniform_K_distr(engine);
      c->Z.push_back(z);
      //update the histograms
      //c->nW[w]++; 
      c->nZ[z]++; 
      add_count(w,z);
    }
    c->shrink_to_fit();

    if(newcell){
      C++;
    }

    c->cell_mutex.unlock();
  }


  void refine(Cell& c){
    if(c.id >=C)
      return;
    refine_count++;
    vector<int> nZg(K); //topic distribution over the neighborhood (initialize with the cell)

    //accumulate topic histogram from the neighbors
    for(auto gid: c.neighbors){
      if(gid <C){
	auto g = get_cell(gid); 
	transform(g->nZ.begin(), g->nZ.end(), nZg.begin(), nZg.begin(), plus<int>());
      }
    }

    transform(c.nZ.begin(), c.nZ.end(), nZg.begin(), nZg.begin(), plus<int>());

    
    vector<double> pz(K,0);

    for(size_t i=0;i<c.W.size(); ++i){
      int w = c.W[i];
      int z = c.Z[i];
      nZg[z]--;
      for(size_t k=0;k<K; ++k){
	int nkw = max<int>(0,nZW[k][w]-1);
	int weight_k = max<int>(0,weight_Z[k]-1); 
	pz[k] = (nkw+beta)/(weight_k + beta*V) * (nZg[k]+alpha);
      } 
      discrete_distribution<> dZ(pz.begin(), pz.end());
      int z_new = min<int>(dZ(engine),K-1);

      nZg[z_new]++;
      relabel(w,z,z_new);
      c.relabel(i,z,z_new);
    } 
  }

  //estimate maximum likelihood topics for the cell
  vector<int> estimate(Cell& c, bool update_ppx=false){
    if(c.id >=C)
      return vector<int>();

    vector<int> nZg(K); //topic distribution over the neighborhood (initialize with the cell)

    //accumulate topic histogram from the neighbors
    for(auto gid: c.neighbors){
      if(gid <C){
	auto g = get_cell(gid); 
	transform(g->nZ.begin(), g->nZ.end(), nZg.begin(), nZg.begin(), plus<int>());
      }
    }
    transform(c.nZ.begin(), c.nZ.end(), nZg.begin(), nZg.begin(), plus<int>());

    int weight_c=0;
    double ppx_sum=0;
    if(update_ppx){ 
      c.perplexity=0;
      weight_c = accumulate(c.nZ.begin(), c.nZ.end(),0);
    }

    vector<double> pz(K,0);
    vector<int> Zc(c.W.size());

    for(size_t i=0;i<c.W.size(); ++i){
      int w = c.W[i];
      int z = c.Z[i];
      nZg[z]--;
      if(update_ppx) ppx_sum=0;

      for(size_t k=0;k<K; ++k){
	int nkw = nZW[k][w];      
	int weight_k = weight_Z[k];
	pz[k] = (nkw+beta)/(weight_k + beta*V) * (nZg[k]+alpha);
	//	if(update_ppx) ppx_sum += pz[k]/(weight_g + alpha*K);
	if(update_ppx) ppx_sum += (nkw+beta)/(weight_k + beta*V) * (c.nZ[k]+alpha)/(weight_c + alpha*K);
      } 
      if(update_ppx)c.perplexity+=log(ppx_sum);
      Zc[i]= max_element(pz.begin(), pz.end()) - pz.begin();
    }
    //if(update_ppx) c.perplexity=exp(-c.perplexity/c.W.size());

    return Zc;
  }

};



