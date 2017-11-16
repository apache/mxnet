/*!
 * Copyright (c) 2015 by Contributors
 * \file profiler.h
 * \brief implements profiler
 */
#ifndef MXNET_ENGINE_PROFILER_H_
#define MXNET_ENGINE_PROFILER_H_

#include <vector>
#include <string>
#include <mutex>
#include <memory>

namespace mxnet {
namespace engine {

/*!
 * \brief Operation execution statistics
 */
struct OprExecStat {
  /*! \brief operation name */
  char opr_name[32];
  /*!
   * \brief operation execution start relative timestamp
   *        time unit is microsecond (10^-6 s)
   */
  uint64_t opr_start_rel_micros;
  /*!
   * \brief operation execution end relative timestamp
   *        time unit is microsecond (10^-6 s)
   */
  uint64_t opr_end_rel_micros;
  /*! \brief id of thread which operation run on */
  uint32_t thread_id;
  /*!
   * \brief device type
   *        CPU: 1, GPU: 2, CPUPinned: 3
   */
  uint32_t dev_type;
  /*! \brief device id */
  uint32_t dev_id;
};

/*!
 * \brief Device statistics
 */
struct DevStat {
  /*! \brief device name */
  std::string dev_name;
  /*! \brief operation execution statistics on this device */
  std::vector<OprExecStat*> opr_exec_stats;
  /*! \brief internal mutex of the execution state */
  std::mutex m_;
};


/*!
 * \brief profiler that records the operation execution information
 *        and saves the profile statistics.
 */
class Profiler {
 public:
  enum ProfilerMode {
      kOnlySymbolic = 0,
      kAllOperator  = 1
  };
  enum ProfilerState {
      kNotRunning = 0,
      kRunning = 1
  };
  /*! \brief set state of profiler */
  void SetState(ProfilerState state);
  /*! \return state of profiler */
  inline ProfilerState GetState() const {
    return this->state_;
  }
  /*! \brief set configure of profiler */
  void SetConfig(ProfilerMode mode, std::string output_filename);
  /*! \return mode of profiler */
  inline ProfilerMode GetMode() const {
    return this->mode_;
  }
  /*! \return whether the profiler is enabled to output */
  inline bool IsEnableOutput() const {
    return this->enable_output_;
  }
  /*! \brief dump the profile file */
  void DumpProfile();
  /*! \return the profiler init time, time unit is microsecond (10^-6) s */
  inline uint64_t GetInitTime() const {
    return init_time_;
  }
  /*! \brief add one operation execution record in
   *   corresponding device statistics */
  OprExecStat* AddOprStat(int dev_type, uint32_t dev_id);
  /*! \return Profiler singleton */
  static Profiler* Get();

 protected:
  /*! \brief make constructor protected. */
  Profiler();

 private:
  /*! \brief generate device information following chrome profile file format */
  void EmitPid(std::ostream *os, const std::string& name, uint32_t pid);
  /*! \brief generate event information following chrome profile file format */
  void EmitEvent(std::ostream *os, const std::string& name,
          const std::string& category, const std::string& ph,
          uint64_t ts, uint32_t pid, uint32_t tid);
  /*! \brief Profiler instance */
  static Profiler* instance_;
  /*! \brief internal mutex of the profiler */
  std::mutex m_;
  /*! \brief indicate whether the profiler is running */
  ProfilerState state_;
  /*! \brief once running, enable profiler to output */
  bool enable_output_;
  /*! \brief indicate what operator the profiler will record */
  ProfilerMode mode_;
  /*! \brief filename to output profile file */
  std::string filename_;
  /*! \brief profile statistics consist of multiple device statistics */
  DevStat* profile_stat;
  /*! \brief cpu number on the machine */
  unsigned int cpu_num_;
  /*! \brief gpu number on the machine */
  unsigned int gpu_num_;
  /*! \brief the profiler init time */
  uint64_t init_time_;
};

/*! \return current clock time, time unit is microsecond (10^-6 s) */
inline uint64_t NowInUsec();
/*! \brief set operation execution start timestamp */
void SetOprStart(OprExecStat* opr_stat);
/*! \brief set operation execution end timestamp */
void SetOprEnd(OprExecStat* opr_stat);

}  // namespace engine
}  // namespace mxnet
#endif  // MXNET_ENGINE_PROFILER_H_
