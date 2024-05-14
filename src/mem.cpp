// Copyright (C) 2021 Chris N. Richardson
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include <chrono>
#include <dolfinx/common/log.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

void process_mem_usage(bool& quit)
{
  const int page_size_bytes = sysconf(_SC_PAGE_SIZE);

  while (!quit)
  {
    std::ifstream f("/proc/self/stat", std::ios_base::in);
    std::istream_iterator<std::string> it(f);
    std::advance(it, 21);

    std::size_t vsize, rss;
    f >> vsize >> rss;
    f.close();
    LOG(WARNING) << "VSIZE=" << vsize / 1024
                 << " RSS=" << rss * page_size_bytes / 1024;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
