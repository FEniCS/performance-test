#include <vector>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <dolfinx/common/log.h>

void process_mem_usage(bool& quit)
{
  const int page_size_bytes = sysconf(_SC_PAGE_SIZE);
    
  while(!quit)
  {
    std::ifstream f("/proc/self/stat", std::ios_base::in);
    std::istream_iterator<std::string> it(f);
    std::advance(it, 21);
    
    std::size_t vsize, rss;  
    f >> vsize >> rss;
    f.close();
    LOG(WARNING) << "VSIZE = " << vsize/1024 << " RSS=" << rss*page_size_bytes/1024 ;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  
}


