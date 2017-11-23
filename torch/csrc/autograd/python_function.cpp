#include "torch/csrc/autograd/python_function.h"

#include <Python.h>
#include <structmember.h>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <ATen/ATen.h>

#include "THP.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/Exceptions.h"

#ifdef WITH_CUDA
#include "cuda/AutoGPU.h"
#endif

using namespace torch;
using namespace torch::autograd;
using namespace torch::jit;
using at::Tensor;

PyObject *THPFunctionClass = NULL;
PyObject *THPBatchNormBackwardBackwardFunction = NULL;

#define THPFunction_assert(condition, ...)                                     \
  if (!(condition)) { THPUtils_setError(__VA_ARGS__); throw python_error(); }

namespace torch { namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
  : type(&var.type())
  , device(-1)
  , size(var.sizes())
  , requires_grad(var.requires_grad()) {
  if (var.type().is_cuda()) {
    device = var.get_device();
  }
}

Variable VariableInfo::zeros(AutoGPU& gpu_guard) const {
  gpu_guard.setDevice(device);
  return type->zeros(size);
}

auto PyFunction::legacy_apply(const variable_list& inputs) -> variable_list {
  AutoGIL gil;

  THPObjectPtr pyInputs(PyTuple_New(inputs.size()));
  if (!pyInputs) throw python_error();

  for (size_t i = 0; i != inputs.size(); ++i) {
    PyObject* input;
    if (inputs[i].defined()) {
      input = createPyObject(inputs[i].data());
      if (!input) throw python_error();
    } else {
      input = Py_None;
      Py_INCREF(input);
    }
    PyTuple_SET_ITEM(pyInputs.get(), i, input);
  }

  THPObjectPtr r(PyObject_CallMethod(
      obj, "_do_backward", "OO", pyInputs.get(), Py_True));
  if (!r) throw python_error();

  auto num_outputs = PyTuple_GET_SIZE(r.get());
  tensor_list tensor_results(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* obj = PyTuple_GET_ITEM(r.get(), i);
    if (obj != Py_None) {
      if (!THPModule_isTensor(obj)) {
        std::string msg("expected Tensor (got '");
        msg += THPUtils_typename(obj);
        msg += "')'";
        throw std::runtime_error(msg);
      }
      tensor_results[i] = createTensor(obj);
    }
  }

  // XXX: this might get requires_grad wrong - there's no way to figure out
  // if _do_backward didn't use ctx.saved_variables and as a result some
  // Variables might require grad, even if no args do. Unfortunately, this
  // leads to unexpected error messages ("no nodes require computing gradients"),
  // but I don't have a better idea. These functions would raise an error
  // in backward anyway.
  return wrap_outputs(inputs, std::move(tensor_results), [this](FunctionFlags &&f) {
    return std::make_shared<Error>(name() + " is not differentiable twice", std::move(f));
  });
}

// NOTE: this function is written in a way that assumes it's only called for backward;
// it's used by engine.cpp.  This is responsible for forwarding a call from
// C++'s Function::apply to a Python method "apply".
auto PyFunction::apply(const variable_list& inputs) -> variable_list {
  AutoGIL gil;
  AutoGPU _gpu_guard(-1);
  THPFunction* py_fn = (THPFunction*)obj;

  THPObjectPtr _legacy(PyObject_GetAttrString(obj, "_is_legacy"));
  if (_legacy == Py_True) {
    return legacy_apply(inputs);
  }

  // Massage a C++ variable_list into a Python arguments tuple
  auto num_inputs = inputs.size();
  THPObjectPtr pyInputs(PyTuple_New(num_inputs));
  if (!pyInputs) throw python_error();
  auto& output_info = py_fn->output_info;
  for (size_t i = 0; i < num_inputs; ++i) {
    PyObject* input;
    if (inputs[i].defined()) {
      input = THPVariable_Wrap(inputs[i]);
    } else {
      input = THPVariable_Wrap(output_info[i].zeros(_gpu_guard));
    }
    if (!input) throw python_error();
    PyTuple_SET_ITEM(pyInputs.get(), i, input);
  }

  THPObjectPtr apply_fn(PyObject_GetAttrString(obj, "apply"));
  if (!apply_fn) throw python_error();
  THPObjectPtr r(PyObject_CallObject(apply_fn, pyInputs.get()));
  if (!r) throw python_error();
  ensure_tuple(r);

  auto& is_variable_input = py_fn->is_variable_input;
  int num_outputs = PyTuple_GET_SIZE(r.get());
  int num_forward_inputs = is_variable_input.size();
  // Returning too many results is ok, but only as long as they're all None.
  // Truncate the result tuple in that case.
  if (num_outputs > num_forward_inputs) {
    bool all_none = true;
    for (int i = num_forward_inputs; i < num_outputs; i++) {
      all_none &= PyTuple_GET_ITEM(r.get(), i) == Py_None;
    }
    if (all_none) {
      num_outputs = num_forward_inputs;
      r = PyTuple_GetSlice(r.get(), 0, num_forward_inputs);
      if (!r) throw python_error();
    }
  }

  // Now the number of gradients should match
  if (num_outputs != num_forward_inputs) {
    std::string msg("function ");
    msg += name() + " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got " ;
    msg += std::to_string(num_outputs) + ")";
    throw std::runtime_error(msg);
  }

  // Massage the Python results tuple back into a C++ variable_list
  variable_list results;
  results.reserve(num_outputs);
  auto& input_info = py_fn->input_info;
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* output = PyTuple_GET_ITEM(r.get(), i);
    bool was_variable = is_variable_input[i];
    if (!was_variable) {
      if (output != Py_None) {
        std::string msg("function ");
        msg += name() + " returned a gradient different than None at position ";
        msg += std::to_string(i + 1) + ", but the corresponding forward input was not a Variable";
        throw std::runtime_error(msg);
      }
      continue;
    }
    if (output == Py_None) {
      auto& info = input_info[results.size()];
      if (info.requires_grad) {
        results.emplace_back(info.zeros(_gpu_guard));
      } else {
        results.emplace_back();
      }
    } else {
      if (!THPVariable_Check(output)) {
        std::string msg("expected Variable or None (got ");
        msg += THPUtils_typename(output);
        msg += ")";
        throw std::runtime_error(msg);
      }
      results.emplace_back(((THPVariable*)output)->cdata);
    }
  }

  return results;
}

auto PyFunction::is_traceable() -> bool {
  AutoGIL gil;
  THPObjectPtr forward_class {PyObject_GetAttrString(obj, "_forward_cls")};
  if (!forward_class) throw python_error();
  THPObjectPtr traceable_py_bool {PyObject_GetAttrString(forward_class, "is_traceable")};
  if (!traceable_py_bool) throw python_error();
  return traceable_py_bool == Py_True;
}

auto PyFunction::releaseVariables() -> void {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  f->saved_variables.clear();
  f->has_freed_buffers = 1;
}

auto PyFunction::name() -> std::string {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  auto name = std::string(Py_TYPE(f)->tp_name);
  THPObjectPtr _legacy(PyObject_GetAttrString(obj, "_is_legacy"));
  if (_legacy == Py_True) {
    name += "LegacyBackward";
  }
  return name;
}

auto PyFunction::getSharedPtr() -> std::shared_ptr<Function> {
  return THPFunction_asFunction((THPFunction*)obj);
}

}} // namespace torch::autograd

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction *self, visitproc visit, void *arg)
{
  for (auto& hook : self->cdata.pre_hooks) {
    if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  for (auto& hook : self->cdata.post_hooks) {
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  Py_VISIT(self->to_save);
  Py_VISIT(self->shared_pairs);
  Py_VISIT(self->non_differentiable);
  Py_VISIT(self->dirty_tensors);
  return 0;
}

static int THPFunction_clear(THPFunction *self)
{
  self->cdata.num_inputs = 0;

  Py_CLEAR(self->needs_input_grad);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->shared_pairs);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);

  self->output_info.clear();
  self->input_info.clear();
  self->saved_variables.clear();
  self->is_variable_input.clear();

  // XXX: this will clear all hooks (not only Python ones)
  // I guess it's ok to leave it as is for now.
  auto pre_hooks = std::move(self->cdata.pre_hooks);
  auto post_hooks = std::move(self->cdata.post_hooks);

  return 0;
}

static void THPFunction_dealloc(THPFunction* self)
{
  PyObject_GC_UnTrack(self);
  THPFunction_clear(self);
  self->cdata.~PyFunction();
  self->output_info.~vector();
  self->input_info.~vector();
  self->saved_variables.~vector();
  self->is_variable_input.~vector();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPFunction_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (!obj) return NULL;
  // Python zero-initializes the object memory, so there's no need to initialize
  // most fields
  THPFunction* self = (THPFunction*)obj;
  new (&self->cdata) PyFunction(obj);
  new (&self->output_info) std::vector<VariableInfo>();
  new (&self->input_info) std::vector<VariableInfo>();
  new (&self->saved_variables) std::vector<SavedVariable>();
  new (&self->is_variable_input) std::vector<bool>();
  self->cdata.num_inputs = -1;
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

using t2var_type = std::unordered_map<PyObject *, THPVariable *>;

// Bump the counters of all recorded dirty input tensors, adding each of them
// into dirty_inputs.  Also does some sanity checking.
static void _mark_dirty(THPFunction *self, t2var_type &t2var,
        std::unordered_set<PyObject *> &dirty_inputs)
{
  // Increase versions of modified tensors
  if (!self->dirty_tensors) return;

  THPFunction_assert(PyTuple_Check(self->dirty_tensors), "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is %s", THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  for (int i = 0; i < num_dirty; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->dirty_tensors, i);
    dirty_inputs.insert(tensor);
    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor), "mark_dirty can "
          "only accept tensors, but argument %d is of type %s", i,
          THPUtils_typename(tensor));
      THPFunction_assert(false, "mark_dirty only accepts input tensors, but "
          "argument %d isn't one", i);
    }
    auto& version_counter = variable->cdata.version_counter();
    version_counter.increment();
  }
  // We're not going to ever need this so let's remove references now
  Py_DECREF(self->dirty_tensors);
  self->dirty_tensors = NULL;
}

// Given a Python tuple of raw output tensors (raw_output), set each of
// the corresponding entries in a different Python tuple (outputs) with
// these tensors wrapped with variables.  We save the gradient function (self)
// to the variable if the output is not volatile (is_volatile).
//
// There is a considerable amount of complexity to handle if the operation
// that produced these output tensors is inplace.  A mapping of *input*
// tensors to variables (t2var) is used to test if this occurred, and
// the set of dirty tensors (dirty_inputs) is used to figure out what to
// do in this case.  After this method is run, t2var is extended with
// mappings for output tensors as well.
static void _wrap_outputs(THPFunction *self, t2var_type &t2var,
    std::unordered_set<PyObject *> &dirty_inputs,
    const t2var_type &shared_pairs,
    PyObject *raw_output, PyObject *outputs, bool is_executable, bool is_volatile)
{
  TORCH_ASSERT(!is_volatile || !is_executable);
  auto cdata = is_executable ? THPFunction_asFunction(self) : nullptr;
  auto flags = VarFlags(is_executable, is_volatile);
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  if (is_executable) {
    self->output_info.clear();
    self->output_info.reserve(num_outputs);
  }

  // Given an output tensor, find the input Variable with which it shares storage
  auto get_shared_base = [&](PyObject* tensor) -> Variable {
    auto input_it = t2var.find(tensor);
    if (input_it != t2var.end()) {
      // If the output is an input treat that as the base
      return input_it->second->cdata;
    }
    auto it = shared_pairs.find(tensor);
    if (it != shared_pairs.end()) {
      // It's explicitly marked as shared via mark_shared_storage
      return it->second->cdata;
    }
    return Variable();
  };

  // Wraps an output Tensor in a Variable or returns the previous wrapper in
  // the case of in-place modification.
  auto wrap_output = [&](at::Tensor data, Variable prev, int output_nr, bool is_modified) -> Variable {
    if (!prev.defined()) {
      return make_variable(std::move(data), flags, output_nr, cdata);
    }
    if (is_modified) {
      if (prev.is_leaf() && prev.requires_grad()) {
        throw std::runtime_error("a leaf Variable that requires grad has been used in an in-place operation.");
      }
      // If the input was modified, transplant the grad_fn in the graph:
      // grad_fn <- variable <- self  ==>  grad_fn <- self <- variable
      prev.get()->grad.reset();
      prev.get()->hooks.clear();
      if (auto grad_acc_fn = prev.get()->grad_accumulator.lock()) {
        auto grad_acc = dynamic_cast<AccumulateGrad*>(grad_acc_fn.get());
        grad_acc->variable.reset();
      }
      prev.rebase_history(flags, output_nr, cdata);
      return prev;
    }
    // An input has been returned, but it wasn't modified. Return it as a view
    // so that we can attach a new grad_fn to the Variable.
    return make_variable_view(std::move(prev), std::move(data), flags, output_nr, cdata);
  };

  t2var_type output2var;
  for (int i = 0; i < num_outputs; i++) {
    PyObject *output = PyTuple_GET_ITEM(raw_output, i);

    THPVariable* output_var;
    auto it = output2var.find(output);
    if (it != output2var.end()) {
      output_var = it->second;
      Py_INCREF(output_var);
    } else {
      // Wrap the output in a Variable
      bool is_modified = dirty_inputs.count(output) > 0;
      Variable var = wrap_output(
          torch::createTensor(output),
          get_shared_base(output),
          i,
          is_modified);

      output_var = (THPVariable*)THPVariable_Wrap(var);
      if (!output_var) throw python_error();

      // We already have the data tensor wrapped as a PyObject*
      Py_INCREF(output);
      Py_CLEAR(output_var->data);
      output_var->data = output;

      output2var[output] = output_var;
    }

    if (is_executable) {
      self->output_info.emplace_back(output_var->cdata);
    }
    PyTuple_SET_ITEM(outputs, i, (PyObject*)output_var);
  }

  // Add every entry in output2var to t2var
  for (auto& entry : output2var) {
    t2var[entry.first] = entry.second;
  }
}

// Save any variables that requested by to_save
static void _save_variables(THPFunction* self, t2var_type &t2var)
{
  if (!self->to_save) return;

  THPFunction_assert(PyTuple_Check(self->to_save), "autograd internal "
      "error: to_save attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->to_save));
  Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
  self->saved_variables.clear();
  self->saved_variables.reserve(num_saved);
  auto cdata_ptr = &self->cdata;
  for (int i = 0; i < num_saved; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->to_save, i);
    if (tensor == Py_None) {
      self->saved_variables.emplace_back();
      continue;
    }

    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch(std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor),
          "save_for_backward can only save tensors, but argument %d is of "
          "type %s", i, THPUtils_typename(tensor));
      THPFunction_assert(false, "save_for_backward can only save input or output "
          "tensors, but argument %d doesn't satisfy this condition", i);
    }

    bool is_output = variable->cdata.grad_fn().get() == cdata_ptr;
    self->saved_variables.emplace_back(variable->cdata, is_output);
  }
  // Free .to_save
  Py_DECREF(self->to_save);
  self->to_save = NULL;
}

// t2var maps input and output tensors to variables
static t2var_type _parse_shared_pairs(THPFunction *self, t2var_type &t2var)
{
  t2var_type map;
  if (!self->shared_pairs) return map;
  THPFunction_assert(PyTuple_Check(self->shared_pairs), "autograd internal "
      "error: shared_pairs attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->shared_pairs));
  Py_ssize_t num_shared = PyTuple_GET_SIZE(self->shared_pairs);
  for (int i = 0; i < num_shared; i++) {
    PyObject *shared_tuple = PyTuple_GET_ITEM(self->shared_pairs, i);
    THPFunction_assert(PyTuple_Check(shared_tuple), "mark_shared_storages "
        "accepts a number of pairs, but one of the arguments is of type %s",
        THPUtils_typename(shared_tuple));
    THPFunction_assert(PyTuple_GET_SIZE(shared_tuple) == 2,
        "mark_shared_storages accepts pairs, but argument %d is a tuple of "
        "%d elements", i, PyTuple_GET_SIZE(shared_tuple));

    // Now we're sure it's really a pair!
    // NB: According to the documentation, v1 is an input tensor, and v2
    // is an output tensor, but we don't actually check this
    PyObject* t1 = PyTuple_GET_ITEM(shared_tuple, 0);
    PyObject* t2 = PyTuple_GET_ITEM(shared_tuple, 1);
    THPFunction_assert(THPModule_isTensor(t1) && THPModule_isTensor(t2),
      "mark_shared_storages accepts pairs of tensors, but one of them "
      "contains %s and %s", THPUtils_typename(t1), THPUtils_typename(t2));

    auto it = t2var.find(t1);
    THPFunction_assert(it != t2var.end(),
        "mark_shared_storages only accepts pairs of input "
        "and output tensors, but argument %d doesn't satify this "
        "condition", i);

    bool inserted;
    std::tie(std::ignore, inserted) = map.emplace(t2, it->second);
    THPFunction_assert(inserted,
        "mark_shared_storages got a duplicate pair for an output tensor at "
        "argument %d", i);
  }
  return map;
}

// Mark requires_grad = 0 on non-differentiable variables (as per non_differentiable)
static void _mark_non_differentiable(THPFunction *self, t2var_type &t2var)
{
  if (!self->non_differentiable) return;

  THPFunction_assert(PyTuple_Check(self->non_differentiable), "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is %s", THPUtils_typename(self->non_differentiable));
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  for (int i = 0; i < num_nondiff; i++) {
    PyObject *t = PyTuple_GET_ITEM(self->non_differentiable, i);
    THPVariable *var;
    try {
      var = t2var.at(t);
      THPFunction_assert(var->cdata.grad_fn().get() == &self->cdata,
          "mark_non_differentiable only accepts output tensors, but "
          "argument %d isn't an output", i);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(t), "mark_non_differentiable "
          "only accepts tensor arguments, but got %s", THPUtils_typename(t));
      THPFunction_assert(false, "mark_non_differentiable only accepts function "
          "outputs");
    }
    var->cdata.requires_grad() = false;
  }
  Py_DECREF(self->non_differentiable);
  self->non_differentiable = NULL;
}

struct UnpackedInput {
  THPObjectPtr tensor_input;
  variable_list input_vars;
};

struct InputFlags {
  FunctionFlags flags;
  THPObjectPtr needs_input_grad;
  std::vector<bool> is_variable_input;
};

template<bool enforce_variables>
std::pair<UnpackedInput, InputFlags> unpack_input(PyObject *args) {
  UnpackedInput unpacked;
  InputFlags flags;

  auto num_args = PyTuple_GET_SIZE(args);
  unpacked.tensor_input = PyTuple_New(num_args);
  flags.needs_input_grad = PyTuple_New(num_args);
  for (int i = 0; i < num_args; i++) {
    PyObject *arg = PyTuple_GET_ITEM(args, i);
    PyObject *new_arg;

    bool is_variable = THPVariable_Check(arg);
    flags.is_variable_input.push_back(is_variable);
    if (!is_variable) {
      if (enforce_variables) {
        THPUtils_setError("expected a Variable argument, but got %s",
                          THPUtils_typename(arg));
        throw python_error();
      }
      Py_INCREF(arg);
      new_arg = arg;
      Py_INCREF(Py_False);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, Py_False);
    } else {
      THPVariable* variable = (THPVariable*)arg;
      new_arg = THPVariable_get_data(variable);
      unpacked.input_vars.push_back(variable->cdata);
      PyObject* needs_grad = variable->cdata.requires_grad() ? Py_True : Py_False;
      Py_INCREF(needs_grad);
      PyTuple_SET_ITEM(flags.needs_input_grad.get(), i, needs_grad);
    }
    PyTuple_SET_ITEM(unpacked.tensor_input.get(), i, new_arg);
  }

  flags.flags = Function::flags(unpacked.input_vars);
  return std::make_pair(std::move(unpacked), std::move(flags));
}

static void _trace_create(PyObject* op_obj, THPFunction* bw_obj,
        PyObject *input_objects, PyObject *output_objects,
        const variable_list& input_vars, bool is_inplace) {
  if (!tracer::isTracing(input_vars))
    return;

  if (!op_obj) {
    std::ostringstream oss;
    oss << "Attempted to trace " << Py_TYPE(bw_obj)->tp_name;
    oss << ", but tracing of legacy functions is not supported";
    throw std::runtime_error(oss.str());
  }

  auto tracing_state = tracer::getTracingState(input_vars);
  bw_obj->is_traced = true;

  // Isolate C variable ptrs in a vector
  variable_list output_vars;
  for (int i = 0; i < PyTuple_GET_SIZE(output_objects); ++i) {
    THPVariable *var = (THPVariable*)PyTuple_GET_ITEM(output_objects, i);
    output_vars.emplace_back(var->cdata);
  }

  // Save scalar args and the calling convention
  auto num_args = PyTuple_GET_SIZE(input_objects);
  pyobj_list scalar_args;
  std::string arg_types;
  arg_types.reserve(num_args);
  scalar_args.reserve(num_args);
  for (int i = 0; i < num_args; i++) {
    PyObject *arg_object = PyTuple_GET_ITEM(input_objects, i);
    if (THPVariable_Check(arg_object)) {
      arg_types.push_back('t');
    } else {
      arg_types.push_back('s');
      Py_INCREF(arg_object);
      scalar_args.emplace_back(arg_object);
    }
  }

  auto state_lock = tracing_state->lock();

  // Note [getValueTrace can allocate nodes]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // When an input variable is not traced, we create a constant instruction
  // to represent it.  This means that you must invoke getValueTrace() BEFORE
  // actually constructing the function that takes these variables as inputs.
  // If we do it the other order, the graph will be in the wrong topological
  // order.

  // See Note [getValueTrace can allocate nodes]
  std::vector<Value*> value_traces;
  value_traces.reserve(input_vars.size());
  for (auto& i : input_vars)
    value_traces.emplace_back(tracer::getValueTrace(tracing_state, i));

  // NB: this function is called only from THPFunction_apply, which is used only
  // when computing forward. All these functions are non-traceable by definition,
  // because they are implemented in terms of tensor operations. Hence, there's no
  // need for any conditionals in here and we can always create the node.

  // Construct the IR Node and its Selects
  Py_INCREF(op_obj);
  auto& graph = tracing_state->graph;
  auto this_expr = graph->appendNode(graph->createPythonOp(
    THPObjectPtr(op_obj),
    arg_types,
    false, // TODO: remove is_legacy
    std::move(scalar_args)));
  for (auto t : value_traces)
    this_expr->addInput(t);

  int num_outputs = output_vars.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = output_vars[i];
    // NOTE: normally we don't add Select nodes when there's only a single
    // output, but Python nodes can't be optimized away, so we simplify the
    // code here.
    auto sel = this_expr->addOutput();
    sel->inferTypeFrom(output.data());
    tracer::setValueTrace(tracing_state, output, sel);
  }
  this_expr->i_(kinplace, is_inplace);

  // See definition in function.cpp.
  THPObjectPtr passes_py_bool {PyObject_GetAttrString(op_obj, "is_traceable")};
  if (!passes_py_bool) throw python_error();
  bool passes_state_transparently = passes_py_bool == Py_True;
  // NB: this path is executed only for forward of Python functions, so there's no need to check
  // tracing_state->in_eval_subgraph (it's always false, because they are never part of backward
  // subgraphs AND we don't even materialize the forward function).
  if (!passes_state_transparently) {
    tracer::nontraceableBackwardSubgraph(input_vars, output_vars);
    Function::setUpContextEdge(this_expr, input_vars, output_vars);
  }
}

PyObject* process_outputs(PyObject *op_obj, THPFunction* grad_fn, const UnpackedInput& unpacked,
                          PyObject *inputs, THPObjectPtr&& raw_output, bool is_executable,
                          bool is_volatile) {
  bool unpack_output = ensure_tuple(raw_output);

  auto num_outputs = PyTuple_GET_SIZE(raw_output.get());

  THPObjectPtr outputs(PyTuple_New(num_outputs));
  if (!outputs) throw python_error();

  grad_fn->cdata.num_inputs = num_outputs;

  // Record type, device, and size information about inputs
  if (is_executable) {
    grad_fn->input_info.clear();
    grad_fn->input_info.reserve(unpacked.input_vars.size());
    for (auto& var : unpacked.input_vars) {
      grad_fn->input_info.emplace_back(var);
    }
  }

  // Initialize t2var map with input tensors
  t2var_type t2var;
  for (auto& c_var : unpacked.input_vars) {
    THPVariable* py_var = (THPVariable*)c_var.get()->pyobj;
    t2var.emplace(py_var->data, py_var);
  }

  std::unordered_set<PyObject *> dirty_inputs;
  bool is_inplace = static_cast<bool>(grad_fn->dirty_tensors);
  _mark_dirty(grad_fn, t2var, dirty_inputs);
  _wrap_outputs(grad_fn, t2var, dirty_inputs,
      _parse_shared_pairs(grad_fn, t2var),
      raw_output, outputs, is_executable, is_volatile);
  // Free shared_pairs
  Py_CLEAR(grad_fn->shared_pairs);
  // At this point, t2var contains output tensors as well
  if (is_executable) {
    _mark_non_differentiable(grad_fn, t2var);
  }
  // NOTE: _trace_create has to run before _save_variables, because we need
  // to assign traces to outputs before we convert them to SavedVariables.
  // On the other hand, it needs to go after _mark_non_differentiable, because
  // it might be wraping backwards in Evals, and _mark_non_differentiable uses
  // grad_fn pointer equality for error checking.
  _trace_create(op_obj, grad_fn, inputs, outputs, unpacked.input_vars, is_inplace);
  if (is_executable) {
    _save_variables(grad_fn, t2var);
  } else {
    // Remove unnecessary attributes
    Py_XDECREF(grad_fn->to_save);
    grad_fn->to_save = NULL;
    Py_XDECREF(grad_fn->non_differentiable);
    grad_fn->non_differentiable = NULL;
  }

  // Unpack the output, unless .forward() returned a tuple
  if (unpack_output) {
    PyObject *output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output);
    return output;
  }

  return outputs.release();
}

// Legacy codepath
PyObject *THPFunction_do_forward(THPFunction *self, PyObject *_inputs)
{
  HANDLE_TH_ERRORS
  torch::autograd::profiler::RecordFunction record(Py_TYPE(self)->tp_name);

  auto info_pair = unpack_input<true>(_inputs);
  auto& unpacked_input = info_pair.first;
  auto& input_info = info_pair.second;
  bool is_executable = input_info.flags.is_executable;
  bool is_volatile = input_info.flags.is_volatile;
  self->cdata.set_flags(std::move(input_info.flags));
  self->needs_input_grad = input_info.needs_input_grad.release();

  // Now we're ready to call a forward (implemented in Python)
  THPObjectPtr forward_fn(PyObject_GetAttrString((PyObject*)self, "forward"));
  if (!forward_fn) return NULL;
  THPObjectPtr raw_output(PyObject_CallObject(forward_fn, unpacked_input.tensor_input));
  if (!raw_output) return NULL;

  return process_outputs(nullptr, self, unpacked_input, _inputs, std::move(raw_output),
                         is_executable, is_volatile);
  END_HANDLE_TH_ERRORS
}

PyObject *THPFunction_apply(PyObject *cls, PyObject *inputs)
{
  HANDLE_TH_ERRORS
  torch::autograd::profiler::RecordFunction record(((PyTypeObject*)cls)->tp_name);

  THPObjectPtr backward_cls(PyObject_GetAttrString(cls, "_backward_cls"));
  if (!backward_cls) return NULL;
  THPObjectPtr ctx_obj(PyObject_CallFunctionObjArgs(backward_cls, NULL));
  if (!ctx_obj) return NULL;
  THPFunction* ctx = (THPFunction*)ctx_obj.get();

  // Prepare inputs and allocate context (grad fn)
  auto info_pair = unpack_input<false>(inputs);
  UnpackedInput& unpacked_input = info_pair.first;
  InputFlags& input_info = info_pair.second;

  // Initialize backward function (and ctx)
  bool is_executable = input_info.flags.is_executable;
  bool is_volatile = input_info.flags.is_volatile;
  ctx->cdata.set_flags(std::move(input_info.flags));
  ctx->needs_input_grad = input_info.needs_input_grad.release();
  ctx->is_variable_input = std::move(input_info.is_variable_input);

  // Prepend ctx to tensor_input, in preparation for static method call
  auto num_args = PyTuple_GET_SIZE(inputs);
  THPObjectPtr ctx_tensor_input(PyTuple_New(num_args + 1));
  PyTuple_SET_ITEM(ctx_tensor_input.get(), 0, ctx_obj.release());
  for (int i = 0; i < num_args; ++i) {
    PyObject *arg = PyTuple_GET_ITEM(unpacked_input.tensor_input.get(), i);
    Py_INCREF(arg);
    PyTuple_SET_ITEM(ctx_tensor_input.get(), i + 1, arg);
  }

  // Call forward
  THPObjectPtr forward_fn(PyObject_GetAttrString(cls, "forward"));
  if (!forward_fn) return NULL;
  THPObjectPtr tensor_outputs(PyObject_CallObject(forward_fn, ctx_tensor_input));
  if (!tensor_outputs) return NULL;

  return process_outputs(cls, ctx, unpacked_input, inputs, std::move(tensor_outputs),
                         is_executable, is_volatile);
  END_HANDLE_TH_ERRORS
}


////////////////////////////////////////////////////////////////////////////////
// Backward
////////////////////////////////////////////////////////////////////////////////

static void _prepare_grad_output(THPFunction *self, THPObjectPtr& raw_grad_output)
{
  AutoGPU gpu_guard(-1);
  int num_grad_output = PyTuple_GET_SIZE(raw_grad_output.get());
  // First, check if any of grad_outputs is None. If not, there's nothing to do
  bool has_none = false;
  for (int i = 0; i < num_grad_output; i++) {
    has_none |= PyTuple_GET_ITEM(raw_grad_output.get(), i) == Py_None;
  }
  if (!has_none)
      return;

  THPObjectPtr grad_output;
  grad_output = PyTuple_New(num_grad_output);
  if (!grad_output) throw python_error();

  // Look for Nones and replace them with new buffers
  auto& output_info = self->output_info;
  for (int i = 0; i < num_grad_output; i++) {
    PyObject *grad = PyTuple_GET_ITEM(raw_grad_output.get(), i);
    if (grad == Py_None) {
      grad = createPyObject(output_info[i].zeros(gpu_guard).data());
      if (!grad) throw python_error();
    } else {
      Py_INCREF(grad);
    }
    PyTuple_SET_ITEM(grad_output.get(), i, grad);
  }
  raw_grad_output = grad_output.release();
}

static void _trim_grad_input(THPFunction *self, THPObjectPtr& grad_input)
{
  int num_grads = PyTuple_GET_SIZE(grad_input.get());
  int num_next_fns = self->cdata.next_functions.size();
  if (num_grads > num_next_fns) {
    // Check that all extra grads are none
    bool all_none = true;
    for (int i = num_next_fns; i < num_grads; i++) {
      all_none = (PyTuple_GET_ITEM(grad_input.get(), i) == Py_None);
      if (!all_none) break;
    }
    // If yes, slice the tuple
    if (all_none) {
      num_grads = num_next_fns;
      grad_input = PyTuple_GetSlice(grad_input.get(), 0, num_grads);
      if (!grad_input) throw python_error();
    }
  }
}

PyObject * THPFunction_do_backward(THPFunction *self, PyObject *args)
{
  try {
    Py_ssize_t num_args = args ? PyTuple_GET_SIZE(args) : 0;
    THPUtils_assert(num_args == 2, "_do_backward expects exactly two arguments");
    PyObject *raw_grad_output = PyTuple_GET_ITEM(args, 0);
    PyObject *retain_variables = PyTuple_GET_ITEM(args, 1);
    if (!PyTuple_Check(raw_grad_output) || !PyBool_Check(retain_variables)) {
      THPUtils_invalidArguments(args, NULL, "_do_backward", 1, "(tuple, bool)");
      return NULL;
    }
    THPUtils_assert(PyTuple_GET_SIZE(raw_grad_output) == self->cdata.num_inputs,
                    "%s got an invalid number of gradients (expected %d got %d)",
                    THPUtils_typename(self), self->cdata.num_inputs,
                    PyTuple_GET_SIZE(raw_grad_output));

    // Some of the output might have been unused, so we have to allocate
    // zero-filled buffers instead
    Py_INCREF(raw_grad_output);
    THPObjectPtr grad_output(raw_grad_output);
    _prepare_grad_output(self, grad_output);

    // self.backward(*grad_output)
    THPObjectPtr backward_fn(PyObject_GetAttrString((PyObject*)self, "backward"));
    THPUtils_assert(backward_fn.get(), "function %s doesn't implement a required "
        "'backward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr grad_input(PyObject_CallObject(backward_fn, grad_output.get()));
    if (!grad_input) return NULL;
    ensure_tuple(grad_input);

    // We allow functions to return more gradients, than there were outputs,
    // if and only if the additional ones are all None
    _trim_grad_input(self, grad_input);
    int num_grads = PyTuple_GET_SIZE(grad_input.get());
    int num_next_fns = self->cdata.next_functions.size();
    THPUtils_assert(num_grads == num_next_fns, "%s returned an invalid number of "
        "gradient tensors (expected %d, but got %d)", THPUtils_typename(self),
        num_next_fns, num_grads);

    return grad_input.release();

  } catch (python_error& e) {
    return NULL;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

PyObject* THPFunction__register_hook_dict(THPFunction *self, PyObject *_var)
{
  THPUtils_assert(THPVariable_Check(_var), "_register_hook_dict expected a variable");
  THPVariable *var = (THPVariable*)_var;
  self->cdata.pre_hooks.emplace_back(new PyFunctionPreHook(var->backward_hooks, var->cdata.output_nr()));
  Py_RETURN_NONE;
}

PyObject* THPFunction_register_hook(THPFunction *self, PyObject *hook)
{
  return torch::autograd::registerFunctionHook(self->cdata, hook);
}

static PyObject *unpack_saved_variables(
    THPFunction *self,
    std::function<PyObject*(const Variable&)> unpack_fn)
{
  THPUtils_assert(!self->has_freed_buffers, ERR_BACKWARD_TWICE);
  auto& saved_variables = self->saved_variables;
  if (saved_variables.empty())
    return PyTuple_New(0);

  int num_saved = saved_variables.size();
  THPObjectPtr saved(PyTuple_New(num_saved));
  if (!saved)
    return NULL;
  auto saved_for = THPFunction_asFunction(self);
  for (int i = 0; i < num_saved; i++) {
    auto unpacked_var = saved_variables[i].unpack(saved_for);
    THPObjectPtr value;
    if (!unpacked_var.defined()) {
      Py_INCREF(Py_None);
      value = Py_None;
    } else {
      value = unpack_fn(unpacked_var);
    }
    PyTuple_SET_ITEM(saved.get(), i, value.release());
  }
  return saved.release();
}

PyObject *THPFunction_saved_tensors(THPFunction *self, void *_unused)
{
  return unpack_saved_variables(self, [](const Variable& var) {
    return createPyObject(var.data());
  });
}

PyObject *THPFunction_saved_variables(THPFunction *self, void *_unused)
{
  return unpack_saved_variables(self, [](const Variable& var) {
    return THPVariable_Wrap(var);
  });
}

PyObject *THPFunction_next_functions(THPFunction *self, void *_unused)
{
  auto& next_fns = self->cdata.next_functions;
  int size = next_fns.size();
  THPObjectPtr result(PyTuple_New(size));
  if (!result)
    return NULL;
  for (int i = 0; i < size; i++) {
    THPObjectPtr fn_tuple(PyTuple_New(2));
    if (!fn_tuple) return NULL;
    PyObject* fn = functionToPyObject(next_fns[i].first);
    if (!fn) return NULL;
    PyTuple_SET_ITEM(fn_tuple.get(), 0, fn);
    PyTuple_SET_ITEM(fn_tuple.get(), 1, PyInt_FromLong(next_fns[i].second));
    PyTuple_SET_ITEM(result.get(), i, fn_tuple.release());
  }
  return result.release();
}


typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

namespace {

template<PyObject* THPFunction::*ptr>
PyObject* getObject(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  PyObject* value = self->*ptr;
  if (!value) {
    Py_RETURN_NONE;
  }
  Py_INCREF(value);
  return value;
}

template<PyObject* THPFunction::*ptr>
int setObject(PyObject* obj, PyObject* value, void* _unused) {
  auto self = (THPFunction*)obj;
  if (value == Py_None) {
    value = nullptr;
  }
  Py_XDECREF((self->*ptr));
  Py_XINCREF(value);
  self->*ptr = value;
  return 0;
}

template<typename M, M THPFunction::*ptr, PyObject* (*Convert)(long)>
PyObject* getMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->*ptr);
}

template<typename M, M Function::*ptr, PyObject* (*Convert)(long)>
PyObject* getImplMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->cdata.*ptr);
}

PyObject* getRequiresGrad(PyObject* obj, void* _unused) {
  Py_RETURN_TRUE;
}

}

static struct PyGetSetDef THPFunction_properties[] = {
  {"saved_tensors", (getter)THPFunction_saved_tensors, NULL, NULL, NULL},
  {"saved_variables", (getter)THPFunction_saved_variables, NULL, NULL, NULL},
  {"next_functions", (getter)THPFunction_next_functions, NULL, NULL, NULL},
  {"to_save", &getObject<&THPFunction::to_save>, &setObject<&THPFunction::to_save>, NULL, NULL},
  {"shared_pairs", &getObject<&THPFunction::shared_pairs>, &setObject<&THPFunction::shared_pairs>, NULL, NULL},
  {"non_differentiable", &getObject<&THPFunction::non_differentiable>, &setObject<&THPFunction::non_differentiable>, NULL, NULL},
  {"dirty_tensors", &getObject<&THPFunction::dirty_tensors>, &setObject<&THPFunction::dirty_tensors>, NULL, NULL},
  {"needs_input_grad", &getObject<&THPFunction::needs_input_grad>, NULL, NULL, NULL},
  {"requires_grad", getRequiresGrad, NULL, NULL, NULL},
  {"_is_tracing", &getMember<char, &THPFunction::is_traced, PyBool_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyMethodDef THPFunction_methods[] = {
  {(char*)"apply", (PyCFunction)THPFunction_apply, METH_CLASS | METH_VARARGS, NULL},
  {(char*)"_do_forward", (PyCFunction)THPFunction_do_forward, METH_VARARGS, NULL},
  {(char*)"_do_backward", (PyCFunction)THPFunction_do_backward, METH_VARARGS, NULL},
  {(char*)"_register_hook_dict", (PyCFunction)THPFunction__register_hook_dict, METH_O, NULL},
  {(char*)"register_hook", (PyCFunction)THPFunction_register_hook, METH_O, NULL},
  {NULL}
};

PyTypeObject THPFunctionType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._FunctionBase",              /* tp_name */
  sizeof(THPFunction),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPFunction_dealloc,       /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPFunction_traverse,    /* tp_traverse */
  (inquiry)THPFunction_clear,            /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPFunction_methods,                   /* tp_methods */
  0,                                     /* tp_members */
  THPFunction_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPFunction_new                        /* tp_new */
};

bool THPFunction_initModule(PyObject *module)
{
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject *)&THPFunctionType);
  return true;
}

struct Decref {
  void operator()(PyFunction* p) const {
    AutoGIL gil;
    Py_DECREF(p->obj);
  }
};

// Similar to shared_from_this. There's a problem that the Python object
// and its cdata depend on each other being alive, so we can't keep
// shared_ptrs as members, but we'd like to be able to manage the lifetime of
// the objects using shared_ptrs in the C++ graph. This returns a new
// shared_ptr, which will decrement the Python reference count when it's
// destructed. WARNING: it's generally not safe to create weak_ptrs from
// these shared_ptrs since multiple shared_ptrs may control the same underlying
// object.
std::shared_ptr<PyFunction> THPFunction_asFunction(THPFunction* self)
{
  if (!self) {
    return std::shared_ptr<PyFunction>();
  }

  Py_INCREF((PyObject*)self);
  return std::shared_ptr<PyFunction>(&self->cdata, Decref());
}
