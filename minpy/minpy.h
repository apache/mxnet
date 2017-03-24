#include <memory>

// TODO(yutian) write C interface in terms of python extension
namespace mxnet {
namespace minpy {

class ImperativeRuntime : public Singleton<ImperativeRuntime> {
 public:
  // Python-side utility functions.
  void EnableJIT() {
    Assert(!jit_enabled_);
    jit_enabled_ = true;
  }
  void DisableJIT() {
    Assert(jit_enabled_);
    FlushJITSequenc();
    jit_enabled_ = false;
  }
  void EnableAutograd() {
    Assert(!autograd_enabled_);
    autograd_enabled_ = true;
  }
  void DisableAutograd() {
    Assert(autograd_enabled_);
    FlushGradSequence();
    autograd_enabled_ = false;
  }
  void StrictEvaluate() {
    DisableAutograd();
    EnableAutograd();
  }

  void Invoke(std::shared_ptr<Symbol> symbol, std::vector<Array> const& inputs,
              std::vector<Array> const& outputs) {
    PushAutogradRecord({symbol, inputs, outputs});
    PushJITRecord({symbol, inputs, outputs});
  }

 private:
  struct ComputingRecord {
    std::shared_ptr<Symbol> symbol;
    std::vector<Array> inputs;
    std::vector<Array> outputs;
  };
  ImperativeRuntime() = default;
  virtual ~ImperativeRuntime() = default;

  void PushAutogradRecord(ComputingRecord record) {
    if (autograd_enabled_) {
      autograd_sequence_.emplace_back(record);
    }
  }
  void PushJITRecord(ComputingRecord record) {
    if (jit_enabled_) {
      // Save for lazy evaluation.
      jit_sequence_.emplace_back(record);
    } else {
      // Strict evaluation.
      engine_->Invoke(record);
    }
  }

  void FlushJITSequence() {
    for (auto&& i : jit_component_.Process(std::move(jit_sequence_))) {
      engine_->Invoke(std::move(i));
    }
    jit_sequence_.clear();
  }

  void FlushGradSequence() {
    for (auto&& i :
         autograd_component_.Process(std::move(autograd_sequence_))) {
      PushJITSequence(std::move(i));
    }
    autograd_sequence_.clear();
  }

  // We own the following two components.
  AutogradComponent autograd_component_{};
  JITComponent jit_component_{};
  std::vector<ComputingRecord> autograd_sequence_{};
  std::vector<ComputingRecord> jit_sequence_{};
  // This is a borrowed reference without ownership.
  std::shared_ptr<MXNetEngine> engine_{};

  bool jit_enabled_ = false;
  bool autograd_enabled_ = false;
};  // class ImperativeRuntime

}  // namespace minpy
}  // namespace mxnet
