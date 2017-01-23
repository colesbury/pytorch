#include <Python.h>
#include <structmember.h>

#include <vector>
#include <unordered_map>
#include <deque>
#include <set>

#include "THP.h"
#include "DynamicTypes.h"
#include <THPP/THPP.h>
#include "./grad_buffer.h"

using thpp::Tensor;
using torch::autograd::Function;
using torch::autograd::GradBuffer;
using variable_list = std::vector<std::shared_ptr<THVariable>>;
using tensor_list = std::vector<std::unique_ptr<Tensor>>;
using function_list = std::vector<std::shared_ptr<Function>>;

PyObject *THPEngineClass = NULL;

// TODO: test if same variable is in backwards multiple times

// used for the queue of nodes ready for processing
using ready_queue_type = std::deque<std::pair<std::shared_ptr<Function>, GradBuffer>>;

std::unordered_map<std::shared_ptr<Function>, int> THPEngine_compute_dependencies(
    function_list& queue,
    ready_queue_type& ready)
{
  std::unordered_map<std::shared_ptr<Function>, int> dependencies;
  std::set<Function*> seen;
  while (queue.size() > 0) {
    auto fn = std::move(queue.back()); queue.pop_back();
    for (auto& prev_fn_pair : fn->previousFunctions()) {
      auto& prev_fn = prev_fn_pair.first;
      if (dynamic_cast<THVariable*>(prev_fn.get()))
        continue;
      // check for stochastic function
      if (prev_fn->isStochastic() && seen.count(prev_fn.get()) == 0 && prev_fn->requiresGrad()) {
        ready.emplace_back(prev_fn, GradBuffer(0));
      } else if (fn->requiresGrad() && prev_fn->requiresGrad()) {
        dependencies[prev_fn] += 1;
      }
      if (seen.count(prev_fn.get()) == 0) {
        seen.insert(prev_fn.get());
        queue.push_back(prev_fn);
      }
    }
  }
  return dependencies;
}

void Engine_backward(const variable_list& variables,
                     tensor_list& grad_variables,
                     bool retain_variables) {
  // OK
  function_list creators;
  ready_queue_type ready;

  bool did_leaf_backward = false;
  int size = variables.size();
  for (int i = 0; i < size; ++i) {
    auto& var = variables[i];
    auto& grad = grad_variables[i];
    if (!var->creator) {
      // If someone calls .backward() on a leaf, it's simple...
      if (var->requires_grad) {
        var->backward(*grad);
        did_leaf_backward = true;
      }
    } else {
      creators.push_back(var->creator);
      if (var->creator->requiresGrad()) {
        GradBuffer buf(var->creator->numOutputs());
        buf.addGrad(var->output_nr, std::move(grad));
        ready.emplace_front(var->creator, std::move(buf));
      }
    }
  }

  auto dependencies = THPEngine_compute_dependencies(creators, ready);

  std::unordered_map<Function*, GradBuffer> not_ready;
  while (ready.size() > 0) {
    auto ready_pair = std::move(ready.back()); ready.pop_back();
    auto& fn = ready_pair.first;
    auto fn_grad_buffers = ready_pair.second.tensors();

    auto grad_inputs = fn->backward(fn_grad_buffers, retain_variables);
    auto previous_functions = fn->previousFunctions();

    if (grad_inputs.size() != previous_functions.size()) {
      throw std::runtime_error("grad_inputs.size() != previous_functions.size()");
    }

    int size = grad_inputs.size();
    for (int i = 0; i < size; ++i) {
      auto& grad_prev = grad_inputs[i];
      auto& prev_fn = previous_functions[i].first;
      int output_nr = previous_functions[i].second;

      if (auto var = dynamic_cast<THVariable*>(prev_fn.get())) {
        if (var->requiresGrad()) {
          var->backward(*grad_prev);
        }
        continue;
      }

      if (prev_fn->isStochastic() || !prev_fn->requiresGrad()) {
        continue;
      }

      // Check if the function is ready for backward
      bool is_ready = false;
      auto it = dependencies.find(prev_fn);
      if (it == dependencies.end()) {
        throw std::runtime_error("dependency not found");
      } else if (--it->second == 0) {
        dependencies.erase(it);
        is_ready = true;
      }

      auto not_ready_it = not_ready.find(prev_fn.get());
      if (is_ready) {
        if (not_ready_it == not_ready.end()) {
          // The function is ready and no buffers have been allocated for it
          GradBuffer prev_buffer(prev_fn->numOutputs());
          prev_buffer.addGrad(output_nr, std::move(grad_prev));
          ready.emplace_front(prev_fn, std::move(prev_buffer));
        } else {
          // The function is ready and it already has a buffer allocated.
          auto prev_buffer = std::move(not_ready_it->second);
          not_ready.erase(not_ready_it);
          prev_buffer.addGrad(output_nr, std::move(grad_prev));
          ready.emplace_front(prev_fn, std::move(prev_buffer));
        }
      } else {
        // Allocate a buffer if necessary and accumulate gradient
        if (not_ready_it == not_ready.end()) {
          GradBuffer prev_buffer(prev_fn->numOutputs());
          prev_buffer.addGrad(output_nr, std::move(grad_prev));
          not_ready.emplace(prev_fn.get(), std::move(prev_buffer));
        } else {
          auto &prev_buffer = not_ready_it->second;
          prev_buffer.addGrad(output_nr, std::move(grad_prev));
        }
      }
    }
  }

  if (!not_ready.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }
}


// Main backward function
PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  PyObject *variables = NULL;
  PyObject *grad_variables = NULL;
  unsigned char retain_variables = 0;
  const char *accepted_kwargs[] = {"variables", "grad_variables",
      "retain_variables", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOb", (char**)accepted_kwargs,
        &variables, &grad_variables, &retain_variables))
    return NULL;
  PyObject *retain_variables_obj = retain_variables ? Py_True : Py_False;

  THPUtils_assert(retain_variables_obj == Py_True || retain_variables_obj == Py_False,
      "retain_variables argument is expected to be a bool, but got %s",
      THPUtils_typename(retain_variables_obj));
  THPUtils_assert(PyTuple_Check(variables), "variables argument is expected to "
      "be a tuple, but got %s", THPUtils_typename(variables));
  THPUtils_assert(PyTuple_Check(grad_variables), "variables argument is "
      "expected to be a tuple, but got %s", THPUtils_typename(grad_variables));

  Py_ssize_t num_variables = PyTuple_GET_SIZE(variables);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_variables);
  THPUtils_assert(num_variables == num_gradients, "got %ld variables and %ld "
      "gradients", num_variables, num_gradients);

  variable_list vars(num_variables);
  tensor_list grads(num_variables);
  for (int i = 0; i < num_variables; i++) {
    PyObject *variable = PyTuple_GET_ITEM(variables, i);
    THPUtils_assert(THPVariable_Check(variable), "element %d of variables "
        "tuple is not a Variable", i);
    vars[i] = *((THPVariable*)variable)->cdata;

    PyObject *grad = PyTuple_GET_ITEM(grad_variables, i);
    if (THPModule_isTensor(grad)) {
      grads[i] = torch::createTensor(grad);
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Tensor or None", i);
    }
  }

  try {
    Engine_backward(vars, grads, retain_variables);
  } catch (python_error &e) {
    return nullptr;
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  Py_RETURN_NONE;
}

PyObject *THPEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  return type->tp_alloc(type, 0);
}

static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward", (PyCFunction)THPEngine_run_backward, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};


PyTypeObject THPEngineType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._EngineBase",                /* tp_name */
  sizeof(THPEngine),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPEngine_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPEngine_new                          /* tp_new */
};


bool THPEngine_initModule(PyObject *module)
{
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject *)&THPEngineType);
  return true;
}
