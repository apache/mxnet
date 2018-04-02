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
 * \file vtune.h
 * \brief VTune API classes.
 */
#ifndef MXNET_PROFILER_VTUNE_H_
#define MXNET_PROFILER_VTUNE_H_

#include <string.h>
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <mshadow/base.h>
#include <atomic>
#include <string>
#include <unordered_map>

#if MXNET_USE_VTUNE
#include <ittnotify.h>
#endif

namespace mxnet {
namespace profiler {
namespace vtune {

MSHADOW_CINLINE void vtune_pause() {
#if MXNET_USE_VTUNE
  __itt_pause();
#endif
}

MSHADOW_CINLINE void vtune_resume() {
#if MXNET_USE_VTUNE
  __itt_resume();
#endif
}

/*! \brief Pause VTune analysis */
struct VTunePause {
  MSHADOW_CINLINE VTunePause() {
    vtune_pause();
  }

  MSHADOW_CINLINE ~VTunePause() {
    vtune_resume();
  }
};

/*! \brief Resume VTune analysis */
struct VTuneResume {
  MSHADOW_CINLINE VTuneResume() {
    vtune_resume();
  }

  MSHADOW_CINLINE ~VTuneResume() {
    vtune_pause();
  }
};

#if MXNET_USE_VTUNE

/*
 * Intel VTune APIs. For API meanings, see:
 * https://software.intel.com/en-us/vtune-amplifier-help-instrumentation-and-tracing-technology-api-reference NOLINT()
 */
class VTuneDomain {
 public:
  inline explicit VTuneDomain(const char *name) throw()
    : domain_(__itt_domain_create(name)) {
    CHECK_NOTNULL(domain_);
    domain_->flags = 1;
  }

  inline operator __itt_domain *() { return domain_; }

  inline __itt_domain *dom() { return domain_; }

  inline const char *name() const { return domain_->nameA; }

 private:
  __itt_domain *domain_;
};

/*!
 * \brief VTune object registry to hold create-once object by name
 * \tparam VTuneObject
 * \note Some objects are expensive to create, such as VTuneEvent, so it's more desirable to
 *       do the mutex lock and reuse
 */
template<typename VTuneObject>
class VTuneRegistry {
 public:
  /*!
   * \brief Retrieve (and create if needed) an object of type 'VTuneObject'
   * \tparam Args Types of arguments to pass to constructor after 'name'
   * \param name Name of the object
   * \param args Arguments to pass to constructor after 'name'
   * \return Pointer to the cached or new VTuneObject
   */
  template<typename ...Args>
  inline VTuneObject *get(const char *name, Args... args) {
    dmlc::ReadLock read_lock(m_);
    auto iter = registry_.find(name);
    if (iter == registry_.end()) {
      dmlc::WriteLock write_lock(m_);
      iter = registry_.emplace(name, std::unique_ptr<VTuneObject>(
        new VTuneObject(name, args...))).first;
    }
    return iter->second.get();
  }

  /*!
   * \brief Retrieve (and create if needed) an object of type 'VTuneObject'
   * \tparam Args Types of arguments to pass to constructor after 'name'
   * \param name Name of the object
   * \param domain Domain of the object (used to create name key)
   * \param args Arguments to pass to constructor after 'domain'
   * \return Pointer to the cached or new VTuneObject
   */
  template<typename ...Args>
  inline VTuneObject *get(const char *name, const VTuneDomain *domain, Args... args) {
    dmlc::ReadLock read_lock(m_);
    auto iter = registry_.find(name);
    if (iter == registry_.end()) {
      dmlc::WriteLock write_lock(m_);
      std::unique_ptr<VTuneObject> ev(new VTuneObject(name, domain, args...));
      iter = registry_.emplace(name, std::unique_ptr<VTuneObject>(
        new VTuneObject(make_key(name, domain), args...))).first;
    }
    return iter->second.get();
  }

 private:
  /*!
   * \brief Make map lookup key given the name and domain of an object
   * \param name Name of the object
   * \param domain Domain of the object
   * \return String key created form the name and domain names
   */
  static inline std::string make_key(const char *name, const VTuneDomain *domain) {
    return std::string(domain->name()) + "::" + std::string(name);
  }

  dmlc::SharedMutex m_;
  std::unordered_map<std::string, std::unique_ptr<VTuneObject>> registry_;
};

/*!
 * \brief VTune Event (per-thread)
 * \note https://software.intel.com/en-us/vtune-amplifier-help-event-api
 * \remark This class has no dependency on the mxnet profiler
 */
class VTuneEvent {
 public:
  inline explicit VTuneEvent(const char *name) throw()
    : itt_event_(__itt_event_create(name, strlen(name))) {}

  inline void start() { __itt_event_start(itt_event_); }

  inline void stop() { __itt_event_end(itt_event_); }

  static VTuneRegistry<VTuneEvent> registry_;

 private:
  __itt_event itt_event_;
};

/*!
 * \brief VTune Task (per-thread, nestable duration)
 * \note https://software.intel.com/en-us/vtune-amplifier-help-task-api
 * \remark This class has no dependency on the mxnet profiler
 */
class VTuneTask {
 public:
  inline VTuneTask(const char *name, VTuneDomain *domain) throw()
    : name_(__itt_string_handle_create(name))
      , domain_(domain) {
  }

  inline void start() { __itt_task_begin(domain()->dom(), __itt_null, __itt_null, name_); }

  inline void stop() { __itt_task_end(domain()->dom()); }

  inline VTuneDomain *domain() { return domain_; }

  const char *name() const { return name_->strA; }

 private:
  __itt_string_handle *name_;
  VTuneDomain *domain_;
};

/*!
 * \brief Frame is a per-process time period with begin and end points (ie video frame)
 * \note https://software.intel.com/en-us/vtune-amplifier-help-frame-api
 * \remark This class has no dependency on the mxnet profiler
 */
class VTuneFrame {
 public:
  inline explicit VTuneFrame(VTuneDomain *domain) throw()
    : domain_(domain) {
    id_ = __itt_id_make(this, 0);
    __itt_id_create(domain->dom(), id_);
  }

  ~VTuneFrame() {
    __itt_id_destroy(domain_->dom(), id_);
  }

#ifdef  MXNET_VTUNE_FRAME_GENERATE_ID
  inline void start() { __itt_frame_begin_v3(domain()->dom(), nullptr); }
  inline void stop()  { __itt_frame_end_v3(domain()->dom(), nullptr); }
#else

  inline void start() { __itt_frame_begin_v3(domain()->dom(), &id_); }

  inline void stop() { __itt_frame_end_v3(domain()->dom(), &id_); }

#endif

  inline operator __itt_id *() { return &id_; }

  inline VTuneDomain *domain() { return domain_; }

 private:
  __itt_id id_;
  VTuneDomain *domain_;
};

/*!
 * \brief VTune Counter object
 * \note https://software.intel.com/en-us/vtune-amplifier-help-frame-api
 * \remark This class has no dependency on the mxnet profiler
 */
class VTuneCounter {
 public:
  inline VTuneCounter(const char *name, VTuneDomain *domain) throw()
    : name_(name)
      , domain_(domain)
      , counter_(__itt_counter_create(name, domain->name())) {
    CHECK_NOTNULL(counter_);
  }

  inline ~VTuneCounter() {
    __itt_counter_destroy(counter_);
  }

  inline void operator++() { __itt_counter_inc_delta(counter_, 1); }

  inline void operator++(int) { __itt_counter_inc_delta(counter_, 1); }

  inline void operator--() { __itt_counter_dec_delta(counter_, 1); }

  inline void operator--(int) { __itt_counter_dec_delta(counter_, 1); }

  inline void operator+=(int64_t v) {
    if (v > 0) {
      __itt_counter_inc_delta(counter_, v);
    } else if (v < 0) {
      __itt_counter_dec_delta(counter_, v);
    }
  }

  inline void operator-=(int64_t v) { this->operator+=(-v); }

  inline VTuneCounter &operator=(uint64_t v) {
    __itt_counter_set_value(counter_, &v);
    return *this;
  }

 private:
  const char *name_;
  VTuneDomain *domain_;
  __itt_counter counter_;
};

/*!
 * \brief VTune single instant-in-time marker
 * \remark This class has no dependency on the mxnet profiler
 */
class VTuneInstantMarker {
 public:
  inline VTuneInstantMarker(const char *name,
                            VTuneDomain *domain,
                            __itt_scope scope = __itt_scope_global) throw()
    : name_(__itt_string_handle_create(name))
      , domain_(domain)
      , scope_(scope) {
  }

  void signal() { __itt_marker(domain_->dom(), __itt_null, name_, scope_); }

 private:
  __itt_string_handle *name_;
  VTuneDomain *domain_;
  __itt_scope scope_;
};

#endif  // MXNET_USE_VTUNE

}  // namespace vtune
}  // namespace profiler
}  // namespace mxnet

#endif  // MXNET_PROFILER_VTUNE_H_
