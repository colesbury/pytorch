#pragma once

namespace torch { namespace autograd {

struct NativeFunction : public Function {
  // virtual ~NativeFunction() {}

  virtual variable_list forward(const variable_list& inputs) = 0;
  virtual tensor_list backward(const tensor_list& gradOutputs, bool retain_variables) = 0;

  // virtual PyObject* pythonObject() override;
  virtual function_list previousFunctions() override {
     return previous_functions;
  }
  virtual int numInputs() const override { return num_inputs; }
  virtual int numOutputs() const override { return num_outputs; }
  virtual bool requiresGrad() const override { return requires_grad; }
  virtual bool isStochastic() const override { return is_stochastic; }

  function_list previous_functions;
  int num_inputs;
  int num_outputs;
  bool requires_grad;
  bool is_stochastic;
  PyObject* python_object; // weak reference
};

}} // namespace torch::autograd
