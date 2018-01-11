/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * \file test_perf.h
 * \brief operator unit test utility functions
 * \author Chris Olivier
*/

#ifndef TEST_PERF_H_
#define TEST_PERF_H_

#ifndef _WIN32
#include <sys/time.h>
#else
#include <Windows.h>
#endif
#include <dmlc/logging.h>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <string>
#include <map>

namespace mxnet {
namespace test {
namespace perf {

/*! \brief current timestamp: millionths of a second */
inline uint64_t getMicroTickCount() {
#ifndef _WIN32
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return uint64_t(tv.tv_sec) * 1000000 + tv.tv_usec;
#else
  LARGE_INTEGER CurrentTime;
  LARGE_INTEGER Frequency;

  QueryPerformanceFrequency(&Frequency);
  QueryPerformanceCounter(&CurrentTime);

  CurrentTime.QuadPart *= 1000000;
  CurrentTime.QuadPart /= Frequency.QuadPart;
  return CurrentTime.QuadPart;
#endif
}

/*! \brief current timestamp: millionths of a second */
inline uint64_t getNannoTickCount() {
#ifndef _WIN32
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t(tv.tv_sec) * 1000000 + tv.tv_usec) * 1000;
#else
  LARGE_INTEGER CurrentTime;
  LARGE_INTEGER Frequency;

  QueryPerformanceFrequency(&Frequency);
  QueryPerformanceCounter(&CurrentTime);

  CurrentTime.QuadPart *= 1000000000;
  CurrentTime.QuadPart /= Frequency.QuadPart;
  return CurrentTime.QuadPart;
#endif
}

#define MICRO2MS(__micro$)  (((__micro$) + 500)/1000)
#define MICRO2MSF(__micro$) (static_cast<float>(__micro$)/1000)
#define MICRO2MSF(__micro$) (static_cast<float>(__micro$)/1000)
#define MS2MICRO(__ms$)     ((__ms$) * 1000)
#define NANO2MSF(__nano$)   (static_cast<float>(__nano$)/1000000)
#define MICRO2S(__micro$)   (((__micro$) + 500000)/1000000)
#define MICRO2SF(__micro$)  (MICRO2MSF(__micro$)/1000)

/*! \brief Calculate time between construction and destruction */
class TimedScope {
  std::string     label_;
  uint64_t        startTime_;
  uint64_t        stopTime_;
  const size_t    count_;

 public:
  explicit inline TimedScope(const char *msg = nullptr, size_t count = 1, const bool start = true)
    : startTime_(start ? getMicroTickCount() : 0)
      , stopTime_(0)
      , count_(count) {
    CHECK_NE(count, 0U);
    if (msg && *msg) {
      label_ = msg;
    }
  }

  explicit inline TimedScope(const std::string& msg, size_t count = 1, const bool start = true)
    : startTime_(start ? getMicroTickCount() : 0)
      , count_(count) {
    CHECK_NE(count, 0U);
    if (!msg.empty()) {
      label_ = msg;
    }
  }

  inline ~TimedScope() {
    print();
  }

  inline void start() {
    startTime_ = getMicroTickCount();
  }

  inline void stop() {
    stopTime_ = getMicroTickCount();;
  }

  inline float elapsedMilliseconds() const {
    const uint64_t elapsed = stopTime_ ? stopTime_ - startTime_ : getMicroTickCount() - startTime_;
    return MICRO2MSF(elapsed);
  }

  inline uint64_t elapsedMicroseconds() const {
    return stopTime_ ? stopTime_ - startTime_ : getMicroTickCount() - startTime_;
  }

  void print() const {
    const uint64_t diff = elapsedMicroseconds();
    std::stringstream ss;
    if (!label_.empty()) {
      ss << label_ << " ";
    }
    ss << "elapsed time: "
       << std::setprecision(4) << std::fixed << MICRO2MSF(diff) << " ms";
    if (count_ != 0 && count_ != 1) {
      const float microSecondsEach = static_cast<float>(diff) / count_;
      ss << " ( " << MICRO2MSF(microSecondsEach) << " ms each )";
    }
    std::cout << ss.str() << std::endl;
  }
};

/*! \brief Accumulate separate timing values mapped by label/id -> total time spent */
class TimingInstrument {
 public:
  explicit TimingInstrument(const char *name = "")
    : name_(name) {
  }
  void startTiming(int id, const char *s) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    auto i = data_.find(id);
    if (i == data_.end()) {
      i = data_.emplace(std::make_pair(id, Info(s))).first;
    }
    if (!i->second.nestingCount_++) {
      i->second.baseTime_ = getMicroTickCount();
    }
  }
  void stopTiming(int id, const size_t subIterationCount = 1) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    auto i = data_.find(id);
    CHECK_NE(i == data_.end(), true) << "Can't stop timing on an object that we don't know about";
    if (i != data_.end()) {
      CHECK_NE(i->second.nestingCount_, 0U) << "While stopping timing, invalid nesting count of 0";
      if (!--i->second.nestingCount_) {
        CHECK_NE(i->second.baseTime_, 0U) << "Invalid base time";
        i->second.duration_.fetch_add(getMicroTickCount() - i->second.baseTime_);
        i->second.baseTime_  = 0;
        i->second.cycleCount_.fetch_add(subIterationCount);
      }
    }
  }
  uint64_t getDuration(int id) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    auto i = data_.find(id);
    if (i != data_.end()) {
      const Info&        info = i->second;
      const uint64_t duration = info.nestingCount_.load()
                                ? info.duration_.load() +
                                  (getMicroTickCount() - info.baseTime_.load())
                                : info.duration_.load();
      return duration;
    }
    return 0;
  }
  bool isTiming(int id) {
    std::unordered_map<int, Info>::const_iterator i = data_.find(id);
    if (i != data_.end()) {
      return i->second.nestingCount_.load() != 0;
    }
    return false;
  }

  template <typename StreamType>
  void print(StreamType *os, const std::string& label_, bool doReset = false) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    // Sorted output
    std::map<int, Info> data(data_.begin(), data_.end());
    for (std::map<int, Info>::const_iterator i = data.begin(), e = data.end();
        i != e; ++i) {
      const Info&        info = i->second;
      const uint64_t duration = getDuration(i->first);
      *os << label_ << ": " << name_ << " Timing [" << info.name_ << "] "
          << (info.nestingCount_.load() ? "*" : "")
          << MICRO2MSF(duration) << " ms";
        if (info.cycleCount_.load()) {
          *os << ", avg: " << (MICRO2MSF(duration) / info.cycleCount_)
              << " ms X " << info.cycleCount_ << " passes";
        }
        *os << std::endl;
    }
    *os << std::flush;
    if (doReset) {
      reset();
    }
  }

  void reset() {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    for (auto i = data_.begin(), e = data_.end();
        i != e; ++i) {
      const int id = i->first;
      const bool wasTiming = isTiming(id);
      if (wasTiming) {
        stopTiming(id);
      }
      // need zero count here
      CHECK_EQ(i->second.nestingCount_.load(), 0U);
      i->second.duration_ = 0;
      if (wasTiming) {
        startTiming(id, i->second.name_.c_str());
      }
    }
  }

  TimingInstrument& operator += (const TimingInstrument& o) {
    for (auto i = o.data_.begin(), e = o.data_.end();
        i != e; ++i) {
      auto j = data_.find(i->first);
      if (j != data_.end())  {
        const Info &oInfo = i->second;
        CHECK_EQ(oInfo.nestingCount_, 0U);
        j->second.duration_   += oInfo.duration_;
        j->second.cycleCount_ += oInfo.cycleCount_;
      } else {
        data_.insert(std::make_pair(i->first, i->second));
      }
    }
    return *this;
  }

  struct Info {
    explicit inline Info(const char *s)
      : name_(s ? s : "")
        , baseTime_(0)
        , nestingCount_(0)
        , cycleCount_(0)
        , duration_(0) {}

    inline Info(const Info& o)
      : name_(o.name_)
        , baseTime_(o.baseTime_.load())
        , nestingCount_(o.nestingCount_.load())
        , cycleCount_(o.cycleCount_.load())
        , duration_(o.duration_.load()) {
      CHECK_EQ(o.nestingCount_, 0U);
    }

    inline Info& operator = (const Info& o) {
      name_ = o.name_;
      baseTime_.store(baseTime_.load());
      nestingCount_.store(nestingCount_.load());
      cycleCount_.store(cycleCount_.load());
      duration_.store(duration_.load());
      return *this;
    }

    /*!
     * \brief Return time for each operation in milliseconds
     * \return Time for each operation in milliseconds
     */
    inline double TimeEach() const {
      return static_cast<double>(duration_) / cycleCount_.load() / 1000.0f;
    }

    std::string           name_;
    std::atomic<uint64_t> baseTime_;
    std::atomic<uint64_t> nestingCount_;
    std::atomic<uint64_t> cycleCount_;  // Note that nesting may skew averages
    std::atomic<uint64_t> duration_;
  };

  typedef std::unordered_map<int, TimingInstrument::Info> timing_map_t;

  const timing_map_t& data() const {
    return data_;
  }

 private:
  std::string                   name_;
  mutable std::recursive_mutex  mutex_;
  std::unordered_map<int, Info> data_;
};

using timing_map_t = TimingInstrument::timing_map_t;

/*! \brief Accumulated scoped timing, indexed by ID */
class TimingItem {
 public:
  inline TimingItem(TimingInstrument *ti,
                    int id,
                    const char *name,
                    const size_t subIterationCount = 1)
    : ti_(ti)
      , id_(id)
      , subIterationCount_(subIterationCount) {
    if (ti_) {
      ti_->startTiming(id, name);
    }
  }
  inline ~TimingItem() {
    if (ti_) {
      ti_->stopTiming(id_, subIterationCount_);
    }
  }

 private:
  TimingInstrument *ti_;
  const int         id_;
  const size_t      subIterationCount_;
};


}  // namespace perf
}  // namespace test
}  // namespace mxnet

#endif  // TEST_PERF_H_
