#ifndef ROST_FLANN_MATCHER
#define ROST_FLANN_MATCHER
#include <opencv/cv.h>
#include "flann/flann.hpp"

class FlannMatcher{
public:
  virtual void set_vocabulary(cv::Mat vocab)=0;
  virtual void get_words(cv::Mat q, vector<int>& words)=0;
};

class BinaryFlannMatcher: public FlannMatcher{
  typedef flann::Hamming<unsigned char> Distance;
  typedef Distance::ElementType ElementType;
  typedef Distance::ResultType DistanceType;
  flann::Matrix<unsigned char> data;
  flann::Matrix<unsigned char> query;
  flann::Matrix<size_t> match;
  flann::Matrix<DistanceType> dists;
  flann::Matrix<DistanceType> gt_dists;
  flann::Matrix<int> indices;
  unsigned int k_nn_;
  flann::Index<Distance> *index;
  flann::IndexParams index_params;
  flann::SearchParams search_params;
public:
  BinaryFlannMatcher():
    index(NULL),
    //    index_params(flann::LinearIndexParams())
    //    index_params(flann::LshIndexParams(12, 20, 2))
    index_params(flann::HierarchicalClusteringIndexParams())    
  {
    //    index_params = flann::HierarchicalClusteringIndexParams();    
  }

  void set_vocabulary(cv::Mat vocab){
    if(index != NULL){
      delete index;
    }
    data = flann::Matrix<unsigned char>(vocab.data, vocab.rows, vocab.cols, vocab.step[0]);
    index = new flann::Index<Distance>(data, index_params);    
    index->buildIndex();
  }

  void get_words(cv::Mat q, vector<int>& words){
    assert(static_cast<int>(q.cols) == static_cast<int>(data.cols));
    query = flann::Matrix<unsigned char>(q.data, q.rows, q.cols, q.step[0]);
    vector<DistanceType> dists_vec(query.rows);
    dists = flann::Matrix<DistanceType>(&(dists_vec[0]), query.rows, 1);
    words.resize(query.rows);
    indices = flann::Matrix<int>(&(words[0]), query.rows, 1);    
    index->knnSearch(query, indices, dists, 1, flann::SearchParams());
  }

  ~BinaryFlannMatcher(){
    if(index != NULL){
      delete index;
    }
  }
};

template<typename Float=float>
class L2FlannMatcher: public FlannMatcher{
  typedef typename flann::L2<Float> Distance;
  typedef typename Distance::ElementType ElementType;
  typedef typename Distance::ResultType DistanceType;
  flann::Matrix<ElementType> data;
  flann::Matrix<ElementType> query;
  flann::Matrix<size_t> match;
  flann::Matrix<DistanceType> dists;
  flann::Matrix<DistanceType> gt_dists;
  flann::Matrix<int> indices;
  flann::Index<Distance> *index;
public:
  L2FlannMatcher():
    index(NULL)
  {
    
  }

  void set_vocabulary(cv::Mat vocab){
    if(index != NULL){
      delete index;
    }
    data = flann::Matrix<ElementType>(reinterpret_cast<ElementType*>(vocab.data), vocab.rows, vocab.cols, vocab.step[0]);
    index = new flann::Index<Distance>(data, flann::LinearIndexParams());    
    index->buildIndex();
  }

  void get_words(cv::Mat q, vector<int>& words){
    query = flann::Matrix<ElementType>(reinterpret_cast<ElementType*>(q.data), q.rows, q.cols, q.step[0]);
    vector<DistanceType> dists_vec(query.rows);
    dists = flann::Matrix<DistanceType>(&(dists_vec[0]), query.rows, 1);
    words.resize(query.rows);
    indices = flann::Matrix<int>(&(words[0]), query.rows, 1);
    index->knnSearch(query, indices, dists, 1, flann::SearchParams());
  }

  ~L2FlannMatcher(){
    if(index != NULL){
      delete index;
    }
  }
};

#endif
