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
using namespace std;


//maps the v to [-N,N)
template<typename T>
T standardize_angle( T v, T N=180){  
  T r = v%(2*N);
  r = r >= N ? r-2*N: r; 
  r = r < -N ? r+2*N: r;
  return r;
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
  vector<vector<int>> nZZ;
  mutex cell_mutex;
  vector<mutex> Z_mutex;
  double perplexity;
  Cell(size_t id_, size_t vocabulary_size, size_t topics_size):
    id(id_),
    //nW(vocabulary_size, 0),
    nZ(topics_size, 0),
    Z_mutex(topics_size),
    nZZ(topics_size, vector<int>(topics_size,0))
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
    //assert(z_old <nZ.size() && z_old >=0);
    //assert(z_new <nZ.size() && z_new >=0);

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

  void update_nZZ(){
    for(size_t i=0;i<nZ.size(); ++i){
      for(size_t j=i;j<nZ.size(); ++j){
	nZZ[i][j] = nZZ[j][i] = nZ[i]*nZ[j];
      }
    }
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
  size_t V, K, C;        //vocab size, topic size, #cells, number of avtive topics
  atomic<int> K_active;
  double alpha, beta, gamma, delta;
  mt19937 engine;
  //ranlux24_base engine;
  //minstd_rand0 engine;

  uniform_int_distribution<int> uniform_K_distr;

  vector<vector<int>> nZW; //nZW[z][w] = freq of words of type w in topic z
  vector<int> weight_Z; //sum_i nZW[z][i]
  vector<vector<int>> nZZ;
  vector<mutex> global_Z_mutex;
  atomic<size_t> refine_count; //number of cells refined till now;
  mutex nZZ_mutex;
  //{...1,1,1,gamma,0,0,..} 
  //gammaZ stores the probability of seeing a give topic
  //if K_active = n, then gammaZ[n]=gamma, gammaZ[0..n-1] = 1, gammaZ[n+1..K-1]=0
  vector<double> gammaZ;  
  bool fixed_K;

  ROST(size_t V_, size_t K_, double alpha_, double beta_, double gamma_, double delta_, const PoseNeighborsT& neighbors_ = PoseNeighborsT(), const PoseHashT& pose_hash_ = PoseHashT()):
    neighbors(neighbors_),
    pose_hash(pose_hash_),
    cell_lookup(1000000, pose_hash),
    V(V_), K(K_), C(0), K_active(K_),
    alpha(alpha_), beta(beta_), gamma(gamma_), delta(delta_),
    uniform_K_distr(),
    //    uniform_K_distr(0,K-1),
    nZW(K,vector<int>(V,0)),
    nZZ(K,vector<int>(K,0)),
    weight_Z(K,0),
    global_Z_mutex(K),
    refine_count(0),
    gammaZ(K,1.0),   //all topics are active by default
    fixed_K(true)
  {
    if(gamma > 0){
      //number of active K grows
      fixed_K=false;
      assert(K>2);
      K_active= 2;
      //disable non-active topics
      for(auto i=K_active.load(); i<K; ++i){
	gammaZ[i]=0;
      }
    }
    assert(K >= K_active);
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
    if(z_old == z_new) return;

    lock(global_Z_mutex[z_new], global_Z_mutex[z_old]);
    
    nZW[z_old][w]--;
    weight_Z[z_old]--;
    nZW[z_new][w]++;
    weight_Z[z_new]++;

    global_Z_mutex[z_old].unlock();
    global_Z_mutex[z_new].unlock();
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
    for(auto w : words){
      c->W.push_back(w);
      //generate random topic label
      int z = uniform_K_distr(engine);
      c->Z.push_back(z);
      //update the histograms
      c->nZ[z]++; 
      add_count(w,z);
    }
    c->shrink_to_fit();

    if(newcell){
      C++;
    }

    c->cell_mutex.unlock();
  }

  void update_nZZ_minus(vector<vector<int>>& diff){
    lock_guard<mutex> lock(nZZ_mutex);
    for(size_t i=0; i< K; ++i){
      for(size_t j=0; j< K; ++j){
	nZZ[i][j]-=diff[i][j];	
      }
    }
  }

  void update_nZZ_plus(vector<vector<int>>& diff){
    lock_guard<mutex> lock(nZZ_mutex);
    for(size_t i=0; i< K; ++i){
      for(size_t j=0; j< K; ++j){
	nZZ[i][j]+=diff[i][j];	
      }
    }
  }

  void daydream(vector<int>& topics){
    vector<int> topics_out(K_active,0);
    vector<discrete_distribution<>> edge_distribution;
    vector<float> pEdge(K_active);
    for(int k=0; k< K_active; ++k){
      for(int i=0;i<K_active; ++i){
	pEdge[i]= nZZ[k][i]+delta;
      }
      discrete_distribution<> edge_dist(pEdge.begin(), pEdge.end());
      for(size_t j=0; j<topics[k]; ++j){
	topics_out[edge_dist(engine)]++;
      }
    }
    topics = topics_out;
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

    if(delta >0){
      daydream(nZg); //smoothen the topic distirbution by daydreaming
    }

    vector<double> pz(K_active+1,0);

    //update topic label for each word in the cell
    for(size_t i=0;i<c.W.size(); ++i){
      int w = c.W[i];
      int z = c.Z[i];
      assert(z>=0 && z < K_active);
      nZg[z]--;

      for(size_t k=0;k<K_active; ++k){
	int nkw = max<int>(0,nZW[k][w]-1);
	int weight_k = max<int>(0,weight_Z[k]-1); 
	pz[k] = (nkw+beta)/(weight_k + beta*V) * (nZg[k]+alpha);
      }

      if(K_active < K){
	pz[K_active]=alpha*gamma/V;
      }

      discrete_distribution<> dZ(pz.begin(), pz.end());
      int z_new = min<int>(dZ(engine), fixed_K ? K-1 : K_active.load() );

      if(z_new == K_active){
	cerr<<"************spawning topic #"<<z_new+1<<endl;
	K_active++;
	pz.resize(K_active+1,0);
      }

      nZg[z_new]++;
      relabel(w,z,z_new);
      c.relabel(i,z,z_new);
    }

    //subtract edge weight counts from the global model
    update_nZZ_minus(c.nZZ);

    //update the topic - topic edge weights for the cell
    c.update_nZZ();

    //add edge weight counts to the global model
    update_nZZ_plus(c.nZZ);
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


template<typename R, typename Stop>
void dowork_parallel_refine_online(R* rost, double tau, int thread_id, Stop stop){
  gamma_distribution<double> gamma1(tau,1.0);
  gamma_distribution<double> gamma2(1.0,1.0);
  size_t now_size = 200;
 
  while(! stop->load() ){
    double r_gamma1 = gamma1(rost->engine), r_gamma2 = gamma2(rost->engine);
    double r_beta = r_gamma1/(r_gamma1+r_gamma2);
    double p_refine_current = generate_canonical<double, 10>(rost->engine);
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

/*
  Word I/O
*/
struct word_reader{
  istream* stream;
  string line;  
  size_t doc_size;
  char delim;
  word_reader(string filename, size_t doc_size_=0, char delim_=' '):
    doc_size(doc_size_),
    delim(delim_)
  {
    if(filename=="-" || filename == "/dev/stdin"){
      stream  = &std::cin;
    }
    else{
      stream = new ifstream(filename.c_str());
    }
  }  
  /*  vector<int> get(){
    vector<int> words;    
    getline(*stream,line);
    if(*stream){
      stringstream ss(line);
      copy(istream_iterator<int>(ss), istream_iterator<int>(), back_inserter(words));
    }
    return words;
    }*/
  vector<int> get(){
    vector<int> words;
    vector<string> words_str;
    string word;
    getline(*stream,line);
    if(*stream){
      //cerr<<"Read line: "<<line<<endl;
      stringstream ss(line);      
      while(std::getline(ss,word,delim)){
	words_str.push_back(word);
      }
      transform(words_str.begin(), words_str.end(), back_inserter(words), [](const string& s){return atoi(s.c_str());});
    }
    return words;
  }
  ~word_reader(){
    if(stream != &std::cin && stream !=NULL){
      delete stream;
      stream=0;
    }
  }
};



