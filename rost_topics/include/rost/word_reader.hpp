#ifndef ROST_WORD_READER_HPP
#define ROST_WORD_READER_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;
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



#endif
