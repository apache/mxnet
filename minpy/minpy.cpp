#include "./minpy.h"

namespace mxnet {
namespace minpy {

namespace {

// Call underlying functin in the old way.
void StrictEvaluation(ComputingRecord record) {
  if (record.delayed_function.index() == 0) {
    PushFCompute(record.delayed_function.get<FCompute>(), record.op,
                 record.attrs, record.ctx, record.read_vars, record.write_vars,
                 record.requested, record.ndinputs, record.ndoutputs);
  } else {
    auto&& p =
        record.delayed_function
            .get<std::pair<std::shared_ptr<Operator>, std::vector<uint32_t>>>();
    PushOperator(p.first, record.op, record.op, record.op, record.op_vars,
                 record.op_vars, record.op, p.second, record.ndinputs,
                 record.ndoutputs);
  }
}

}  // anonymous namespace

void ImperativeRuntime::EnableJIT() {
  Assert(!jit_enabled_);
  jit_enabled_ = true;
}

void ImperativeRuntime::DisableJIT() {
  Assert(jit_enabled_);
  FlushJITSequenc();
  jit_enabled_ = false;
}

void ImperativeRuntime::EnableAutograd() {
  Assert(!autograd_enabled_);
  autograd_enabled_ = true;
}

void ImperativeRuntime::DisableAutograd() {
  Assert(autograd_enabled_);
  FlushGradSequence();
  autograd_enabled_ = false;
}

void ImperativeRuntime::StrictEvaluate() {
  DisableAutograd();
  EnableAutograd();
  // TODO(yutian): Call `MXNDArrayWaitToRead`
}

void ImperativeRuntime::Invoke(ComputingRecord record) {
  PushAutogradRecord(record);
  PushJITRecord(record);
}

void ImperativeRuntime::PushAutogradRecord(ComputingRecord record) {
  if (autograd_enabled_) {
    autograd_seqeunce_.emplace_back(std::move(record));
  }
}

void ImperativeRuntime::PushJITRecord(ComputingRecord record) {
  if (jit_enabled_) {
    // Save for lazy evaluation.
    jit_sequence_.emplace_back(std::move(record));
  } else {
    StrictEvaluate(std::move(record));
  }
}

void ImperativeRuntime::FlushJITSequence() {
  for (auto&& i : jit_component_.Process(std::move(jit_sequence_))) {
    StrictEvaluate(std::move(i));
  }
  jit_sequence_.clear();
}

void ImperativeRuntime:: ::FlushGradSequence() {
  for (auto&& i : autograd_component_.Process(std::move(autograd_sequence_))) {
    PushJITSequence(std::move(i));
  }
  autograd_sequence_.clear();
}

}  // namespace minpy
}  // namespace mxnet
