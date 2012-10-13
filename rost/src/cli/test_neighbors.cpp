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
  neighbors<Pose>g2(Pose{{1,2,3}});
  for(auto g : g2(a)){
    cout<<g<<endl;
  }

  return 0;
}
