#pragma once

#include <Python.h>
#include <mutex>
#include <memory>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct VariableTensor : public at::TensorImpl {
public:
  explicit VariableTensor(at::Tensor data);
  VariableTensor(at::Tensor data, std::shared_ptr<Function> grad_fn);
  virtual ~VariableTensor();
  virtual const char * toString() const override;
  virtual at::IntList sizes() override;
  virtual at::IntList strides() override;
  virtual int64_t dim() override;
  virtual at::Scalar localScalar() override;
  virtual void assign_(at::Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  static const char * typeString();

public:
  std::shared_ptr<Function> get_grad_accumulator();

  at::Tensor data;
  at::Tensor grad;
  std::shared_ptr<Function> grad_fn;
  std::unique_ptr<VariableVersion> version_counter;
  std::vector<std::shared_ptr<FunctionPreHook>> hooks;
  std::weak_ptr<Function> grad_accumulator;
  std::mutex grad_accumulator_lock;
  bool requires_grad;
  bool is_volatile;
  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  int output_nr;
  PyObject *pyobj;  // weak reference

  friend struct VariableType;
};

}} // namespace torch::autograd
