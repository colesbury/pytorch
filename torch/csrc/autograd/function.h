#ifndef THP_FUNCTION_H
#define THP_FUNCTION_H

struct THPFunction;

class python_error : public std::exception {};

struct THPFunctionPtr: public THPObjectPtr {
    THPFunctionPtr(): THPObjectPtr(nullptr), output_nr(-1) {};

    THPFunctionPtr(PyObject *fn, int output_nr):
        THPObjectPtr(fn), output_nr(output_nr) {};

    THPFunctionPtr(THPFunction *fn, int output_nr):
        THPObjectPtr((PyObject*)fn), output_nr(output_nr) {};

    THPFunctionPtr(THPFunctionPtr &&other):
        THPObjectPtr(std::move(other)), output_nr(other.output_nr) {}

    THPPointer& operator =(THPFunctionPtr &&other) {
        output_nr = other.output_nr;
        THPObjectPtr::operator=(std::move(other));
        return *this;
    }

    int output_nr;
};

struct THPVariableVersion;
struct THVariable;

namespace thpp {
  struct Tensor;
}

namespace torch { namespace autograd {

struct ExecutionContext {
  PyThreadState* thread_state;
};

struct Function {
  using tensor_list = std::vector<std::unique_ptr<thpp::Tensor>>;
  using variable_list = std::vector<std::shared_ptr<THVariable>>;
  using function_list = std::vector<std::pair<std::shared_ptr<Function>, int>>;

  Function() {};
  Function(const Function& other) = delete;
  Function(Function&& other) = delete;
  virtual ~Function() {};

  // virtual void forward(const variable_list& inputs) = 0;
  virtual tensor_list backward(const tensor_list& gradOutputs, bool retain_variables) = 0;
  virtual PyObject* pythonObject() = 0;

  virtual function_list previousFunctions() = 0;
  virtual int numInputs() const = 0;
  virtual int numOutputs() const = 0;
  virtual bool requiresGrad() const = 0;
  virtual bool isStochastic() const = 0;
};

struct PyFunctionWrapper : public Function {
  PyFunctionWrapper(PyObject *obj);
  virtual ~PyFunctionWrapper();

  virtual tensor_list backward(const tensor_list& gradOutputs, bool retain_variables) override;
  virtual PyObject* pythonObject() override;

  virtual function_list previousFunctions() override;
  virtual int numInputs() const override;
  virtual int numOutputs() const override;
  virtual bool requiresGrad() const override;
  virtual bool isStochastic() const override;

private:
  THPObjectPtr pyobj;
};


}} // namespace torch::autograd

// (class, gpu id, sizes)
using output_info_type = std::tuple<PyObject *, int, std::vector<long>>;
// (tensor, version when saved, version counter)
// or
// (None, 0, nullptr)
using saved_var_info_type = std::tuple<THPObjectPtr, int, std::unique_ptr<THPVariableVersion>>;

struct THPFunction {
    PyObject_HEAD

    PyObject *needs_input_grad;
    PyObject *backward_hooks;
    THPObjectPtr *output_backward_hooks;

    PyObject *to_save;
    PyObject *shared_pairs;
    PyObject *non_differentiable;
    PyObject *dirty_tensors;

    std::weak_ptr<torch::autograd::PyFunctionWrapper> wrapper;

    THPFunctionPtr *previous_functions;
    std::vector<output_info_type> *output_info;
    std::vector<saved_var_info_type> *saved_variables;
    int num_inputs;
    int num_outputs;
    char requires_grad;
    char has_freed_buffers;
};

bool THPFunction_initModule(PyObject *module);
extern PyObject *THPFunctionClass;
extern PyObject *THPStochasticFunctionClass;

std::shared_ptr<torch::autograd::PyFunctionWrapper> THPFunction_asFunction(THPFunction* self);

#define THPFunction_Check(obj) PyObject_IsInstance(obj, THPFunctionClass)

#endif
