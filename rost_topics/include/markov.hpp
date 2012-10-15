#ifndef MARKOV_HPP
#define MARKOV_HPP
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <vector>
//namespace entropy2{
using namespace std;
using boost::uniform_real;
using boost::uniform_int;

template<typename T>
class Stirling{
public:
  vector<vector<T> > cache;
  Stirling():cache(1,vector<T>(1,T(1))){
  }
  T operator()(unsigned int N, unsigned int m){
    if(N==0 && m==0) return T(1);
    if(m==0 || m > N) return T(0);
    //increase the cache size if needed
    ensure_cache_till(N);
    return cache[N][m];
  }

  void ensure_cache_till(unsigned int N){
    if (N < cache.size())
      return;
    for(size_t ni=cache.size(); ni<=N; ++ni){
      cache.push_back(vector<T>(ni+1));
      cache[ni][0]=T(0);
      cache[ni][ni]=T(1);
      for(size_t mi=1;mi<ni; ++mi){
	cache[ni][mi] = cache[ni-1][mi-1] + (ni-1)*cache[ni-1][mi];
      }
    } 
  }
  T* operator()(unsigned int N){
    //increase the cache size if needed
    ensure_cache_till(N);
    return &(cache[N][0]);
  }
};


template<typename Float=float,typename Int=int>
class Random{
public:

  typedef boost::variate_generator<boost::mt19937&, boost::uniform_real<Float> > die_01_t;
  typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<Int> > die_t;

  boost::mt19937 randgen;
  uniform_real<Float> uniform_01_dist;
  die_01_t die_01;
  Stirling<double> stirling;


  Random():uniform_01_dist(0,1),
	   die_01(randgen,uniform_01_dist)
  {
  }

  Int die(int N){ //a random number between [0..N-1]
    int r = die_01()*N;
    return r;
  }

  Int integer(int N){ //a random number between [0..N-1]
    int r = die_01()*N;
    assert(r==0 || r<N);
    return r;
  }
    
  template <typename It>
  void uniform_sample(It begin, It end, size_t K){
    die_t die(randgen, uniform_int<Int>(0,static_cast<Int>(K-1)));
    for(; begin!=end; ++begin){
      *begin=die();
    }
  }

  size_t category_sample(const std::vector<float>&p, float total_p){
    Float r = die_01();
    Float u=r*total_p; //normalize it 
    
    total_p=0;
    size_t k;
    for(k=0;k<p.size();k++){
      total_p+=p[k];
      if(u<=total_p) break;
    }
    if(k==p.size()) k--;
     return k;
  }

  template<typename It>
  size_t category_sample_from_pdf(It begin, It end){
    typename iterator_traits<It>::value_type total=0;
    total = accumulate(begin, end,0);
    size_t k=0;
    double u=die_01()*total;
    total=0;
    for(It i=begin; i!=end; ++i){
      total+=*i;
      if(u<=total)
	return k;
      k++;
    }
    cerr<<"ERROR: something wrong with category sampling.\n";
    assert(false);//we should never end up being here
    return k;
  }

  //computes a random sample from the given cdf
  //*(end-1) is assumed to be the total mass
  //uses binary search to speed up things
  template<typename It>
  size_t category_sample_from_cdf(It begin, It end){
    double u=die_01() * (*(end -1));
    return lower_bound(begin,end,u) - begin;
  }

  template<typename T>
  T beta(Float a, Float b){
    return boost::math::cdf(boost::math::beta_distribution<T>(a,b), die_01());
  }

  
  template<typename T>
  T gamma(T shape){
    boost::gamma_distribution<T> gamma_dist(shape);
    boost::variate_generator<boost::mt19937&,boost::gamma_distribution<T> > gamma_gen(randgen, gamma_dist );
    return gamma_gen();

    //return boost::math::cdf(boost::math::gamma_distribution<T>(a), die_01());
  }

  /// generate a sample from a DirichletDistribtion(alpha)
  /// alpha_begin, alpha_end are the iterators for alpha vector
  /// output is written to out iterator. 
  template<typename InIt, typename OutIt>
  OutIt dirichlet(InIt alpha_begin, InIt alpha_end, OutIt out){
    vector<typename iterator_traits<InIt>::value_type> samples(alpha_end - alpha_begin);
    typename vector<typename iterator_traits<InIt>::value_type>::iterator si, si_end;
    si = samples.begin(); si_end=samples.end();

    typename iterator_traits<InIt>::value_type total(0), v;
    for(InIt i=alpha_begin; i!=alpha_end; ++i){
      v=gamma<typename iterator_traits<InIt>::value_type >(*i);
      *si++ = v;
      total+=v;
    }
    for(si=samples.begin(); si!=si_end; ++si){
      *out++ = *si/total;
    }
    return out;
  }

  /// sample the number of components that a DirichletProcess(alpha) has after n samples. 
  /// P(m | alpha, n)  = stirling(n,m) alpha^m Gamma(alpha)/Gamma(alpha+n)
  /// (Antoniak 1974)
  /// we first compute the above distribution, ignoring the gamma constant and then extract a sample
  unsigned int antoniak_sample(double alpha, unsigned int n){
    assert(n>0);
    double * s = stirling(n);
    vector<double> cdf_m(n+1);
    double alpha_to_m=alpha; //alpha^1 = alpha
    cdf_m[0]=0;
    for(size_t m=1;m<=n; ++m){
      cdf_m[m]=cdf_m[m-1]+s[m]*alpha_to_m;
      alpha_to_m *= alpha;
    }
    return category_sample_from_cdf(cdf_m.begin(), cdf_m.end());
  }
};

template<typename T>
vector<int> histogram(const vector<T>& values, unsigned int K){
  vector<int> hist(K,0);
  if(values.empty())
    return hist;
  
  //  K = max(*(max_element(values.begin(), values.end())),K);
  
  //  hist.resize(K,0);
  for(size_t i=0;i<values.size();++i){
    if(values[i]>= K)
      cerr<<"values[i]="<<values[i]<<"  K="<<K<<endl;
    assert(values[i]<K);
    hist[values[i]]++;
  }
  return hist;
}

template<typename T>
vector<float> normalize(const vector<T>& histogram, float alpha=0.0){
  vector<float> nhist(histogram.size());
  float total=0;
  for(size_t i=0; i<histogram.size(); ++i){
    total+=(histogram[i]+alpha);
  }
  for(size_t i=0; i<histogram.size(); ++i){
    nhist[i]=(static_cast<float>(histogram[i])+alpha)/total;
  }
  return nhist;
}

template<typename T>
double entropy(const vector<T>& histogram){
  T total=0;
  for(size_t i=0; i<histogram.size(); ++i){
    total+=histogram[i];
  }
  double h=0, pi;
  for(size_t i=0; i<histogram.size(); ++i){
    pi=static_cast<double>(histogram[i])/static_cast<double>(total);
    h-=pi*log(pi);
  }
  return h;
}
//returns a histogram of frequency counts of discrete values in the given vector
//K+1 is the number of bins in the returned histogram
//if not specified, K is the maximum value in the the given vlues vector
/*vector<int> histogram(const vector<int>& values, int K=0);

void histogram(const std::vector<int>&values, std::vector<int>& bins);
//void histogram(const cv::Mat_<float>& values, cv::Mat_<float>& bins);

vector<float> normalize(const vector<int>& histogram,float alpha=0);
float entropy(const vector<int>& histogram);
float entropy(const vector<float>& normalized_histogram);
*/



#endif
