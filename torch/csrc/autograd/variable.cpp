#include <Python.h>
#include <structmember.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/cuda/AutoGPU.h"
#include "torch/csrc/autograd/native_function.h"
#include "torch/csrc/autograd/python_native_function.h"


using namespace torch;
using namespace thpp;


PyObject *THPVariableClass = NULL;

THVariable::THVariable(
  std::unique_ptr<thpp::Tensor> data,
  char requires_grad,
  char is_volatile)
    : data(std::move(data))
    , creator(nullptr)
    , grad(nullptr)
    , version_counter(new THPVariableVersion())
    , output_nr(0)
    , is_volatile(is_volatile)
    , requires_grad(requires_grad)
    , backward_hooks(nullptr)
    , pyobj(nullptr)
{
  if (!this->data) {
    throw std::runtime_error("Variable data is NULL");
  }
}

THVariable::THVariable(
  std::unique_ptr<thpp::Tensor> data,
  std::shared_ptr<torch::autograd::NativeFunction> creator)
    : data(std::move(data))
    , creator(creator)
    , grad(nullptr)
    , version_counter(new THPVariableVersion())
    , output_nr(creator->num_outputs++)
    , is_volatile(creator->is_volatile)
    , requires_grad(creator->requires_grad)
    , backward_hooks(nullptr)
    , pyobj(nullptr)
{
  if (!this->data) {
    throw std::runtime_error("Variable data is NULL");
  }
}

bool THVariable::is_cuda()
{
  return data->isCuda();
}

TensorType THVariable::tensor_type()
{
  TensorType type;
  type.data_type = data->type();
  type.is_cuda = data->isCuda();
  type.is_sparse = false;
  return type;
}

static PyObject* THPVariable_New3(PyTypeObject* type, std::shared_ptr<THVariable>& var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) std::shared_ptr<THVariable>(var);
  }
  return obj;
}

PyObject * THPVariable_Wrap(std::shared_ptr<THVariable>& var)
{
  if (var->pyobj) {
    Py_INCREF(var->pyobj);
  } else {
    var->pyobj = THPVariable_New3((PyTypeObject *)THPVariableClass, var);
  }
  return var->pyobj;
}

PyObject * THPVariable_New2(PyTypeObject *type, PyObject *data, PyObject *creator, char requires_grad, char is_volatile)
{
  THPUtils_assert(THPModule_isTensor(data), "data must be a Tensor");
  THPUtils_assert(!creator || THPFunction_Check(creator), "creator must be a Function");
  auto v = std::make_shared<THVariable>(createTensor(data), requires_grad, is_volatile);
  PyObject* obj = THPVariable_New3(type, v);
  if (obj) {
    v->pyobj = obj;
    v->creator = THPFunction_asFunction((THPFunction*)creator);
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}


// This function DOES NOT steal a reference to data and creator
// To create a leaf Variable pass NULL as creator.
PyObject * THPVariable_New(PyObject *data, PyObject *creator, char requires_grad, char is_volatile)
{
  THPUtils_assert(THPModule_isTensor(data), "data must be a Tensor");
  PyTypeObject *type = (PyTypeObject *)THPVariableClass;
  return THPVariable_New2(type, data, creator, requires_grad, is_volatile);
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewVolatile(PyObject *data)
{
  return THPVariable_New(data, nullptr, 0, 1);
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->data);
  auto& var = *self->cdata;
  // Py_VISIT(var.creator);
  Py_VISIT(var.backward_hooks);
  return 0;
}

static int THPVariable_clear(THPVariable *self)
{
  Py_CLEAR(self->data);
  auto& var = *self->cdata;
  // Py_CLEAR(var.creator);
  Py_CLEAR(var.backward_hooks);
  return 0;
}

static void THPVariable_dealloc(THPVariable* self)
{
  using std::shared_ptr;
  PyObject_GC_UnTrack(self);
  Py_XDECREF(self->data);
  auto& var = *self->cdata;
  // Py_XDECREF(var.creator);
  Py_XDECREF(var.backward_hooks);
  var.pyobj = nullptr;
  self->cdata.~shared_ptr<THVariable>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  PyObject *data;
  PyObject *creator = NULL;
  char is_volatile = 0;
  char requires_grad = 0;

  const char *accepted_args[] = {"data", "creator", "volatile", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Obb", (char**)accepted_args,
      &data, &creator, &is_volatile, &requires_grad))
    return NULL;

  if (creator == Py_None)
    creator = NULL;

  THPUtils_assert(!(is_volatile && requires_grad),
          "Variable can't be volatile and require_grad at the same time!");
  THPUtils_assert(!creator || THPFunction_Check(creator),
          "Variable creator has to be a Function object or None, but got %s",
          THPUtils_typename(creator));
  THPUtils_assert(THPModule_isTensor(data), "Variable data has to "
          "be a tensor, but got %s", THPUtils_typename(data));

  return THPVariable_New2(type, data, creator, requires_grad, is_volatile);
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_version(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyInt_FromLong(**var.version_counter);
}

PyObject *THPVariable_get_creator(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.creator) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.creator);
}

PyObject * THPVariable_get_data(THPVariable *self)
{
  if (!self->data) {
    auto& var = *self->cdata;
    PyTypeObject* type = getPyTypeObject(*var.data);
    self->data = type->tp_alloc(type, 0);
    if (self->data) {
      ((THPVoidTensor*)self->data)->cdata = (THVoidTensor *)var.data->retain().cdata();
    }
  }
  Py_INCREF(self->data);
  return self->data;
}

int THPVariable_set_data(THPVariable *self, PyObject *data)
{
  THPUtils_assertRet(-1, THPModule_isTensor(data), "Variable data has to "
      "be a tensor, but got %s", THPUtils_typename(data));
  Py_INCREF(data);
  Py_XDECREF(self->data);
  self->data = data;
  auto& var = *self->cdata;
  auto tensor = createTensor(data);
  var.data.swap(tensor);
  return 0;
}

PyObject *THPVariable_get_raw_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad) {
    Py_RETURN_NONE;
  }
  return THPVariable_Wrap(var.grad);
}

int THPVariable_set_raw_grad(THPVariable *self, PyObject *data)
{
  auto& var = *self->cdata;
  if (data == Py_None) {
    var.grad.reset();
    return 0;
  }
  THPUtils_assertRet(-1, THPVariable_Check(data),
      "expected Variable or None (got %s)", THPUtils_typename(data));
  var.grad = ((THPVariable*)data)->cdata;
  return 0;
}

PyObject *THPVariable_get_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad) {
    THCPAutoGPU __guard(var.data->getDevice());
    auto grad = var.data->newTensor();
    grad->resizeAs(*var.data).zero();
    var.grad = std::make_shared<THVariable>(std::move(grad), 0, 1);
  }
  return THPVariable_Wrap(var.grad);
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyBool_FromLong(var.is_volatile);
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "volatile must be a bool");
  auto& var = *self->cdata;
  var.is_volatile = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_output_nr(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyInt_FromLong(var.output_nr);
}

PyObject *THPVariable_get_requires_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyBool_FromLong(var.requires_grad);
}

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "requires_grad must be a bool");
  auto& var = *self->cdata;
  if (var.creator) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  var.requires_grad = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self)
{
  auto& var = *self->cdata;
  if (var.backward_hooks) {
    Py_INCREF(var.backward_hooks);
    return var.backward_hooks;
  }
  Py_RETURN_NONE;
}

int THPVariable_set_backwards_hooks(THPVariable *self, PyObject *obj)
{
  auto& var = *self->cdata;
  Py_INCREF(obj);
  Py_XDECREF(var.backward_hooks);
  var.backward_hooks = obj;
  return 0;
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_version", (getter)THPVariable_get_version, NULL, NULL, NULL},
  {"creator", (getter)THPVariable_get_creator, NULL, NULL, NULL},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, NULL, NULL},
  {"_grad", (getter)THPVariable_get_raw_grad, (setter)THPVariable_set_raw_grad, NULL, NULL},
  {"grad", (getter)THPVariable_get_grad, NULL, NULL, NULL},
  {"volatile", (getter)THPVariable_get_volatile, (setter)THPVariable_set_volatile, NULL, NULL},
  {"output_nr", (getter)THPVariable_get_output_nr, NULL, NULL, NULL},
  {"requires_grad", (getter)THPVariable_get_requires_grad, (setter)THPVariable_set_requires_grad, NULL, NULL},
  {"_backward_hooks", (getter)THPVariable_get_backwards_hooks, (setter)THPVariable_set_backwards_hooks, NULL, NULL},
  {NULL}
};


PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableBase",              /* tp_name */
  sizeof(THPVariable),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPVariable_dealloc,       /* tp_dealloc */
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
  (traverseproc)THPVariable_traverse,    /* tp_traverse */
  (inquiry)THPVariable_clear,            /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPVariable_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPVariable_pynew                      /* tp_new */
};


bool THPVariable_initModule(PyObject *module)
{
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_VariableBase", (PyObject *)&THPVariableType);
  return true;
}

auto THVariable::backward(const Tensor& _gradOutput) -> void {
  std::unique_ptr<Tensor> modified_grad;
  const Tensor* gradOutput = &_gradOutput;
  if (backward_hooks) {
    // FIXME: GIL
    THPObjectPtr grad = createPyObject(*gradOutput);
    if (!grad) throw python_error();

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(backward_hooks, &pos, &key, &value)) {
      THPObjectPtr res = PyObject_CallFunctionObjArgs(value, grad.get(), nullptr);
      if (!res) throw python_error();
      grad = std::move(res);
    }

    modified_grad = std::move(createTensor(grad.get()));
    gradOutput = modified_grad.get();
  }
  if (!grad) {
    std::unique_ptr<thpp::Tensor> copy(gradOutput->clone());
    grad.reset(new THVariable(std::move(copy), (char)0, (char)1));
  } else {
    grad->data->cadd(*grad->data, *gradOutput);
  }
}

auto THVariable::apply(const variable_list& gradOutputs) -> variable_list {
  if (creator || **version_counter != 0) {
    throw std::runtime_error("leaf variable was used in an inplace operation");
  }
  if (gradOutputs.size() != 1) {
    throw std::runtime_error("incorrect number of gradOutputs");
  }
  backward(*gradOutputs[0]->data);
  return variable_list();
}

auto THVariable::previousFunctions() -> function_list {
  if (creator) {
    return function_list({ std::make_pair<>(creator, output_nr) });
  }
  return function_list();
}

auto THVariable::numOutputs() const -> int {
  return 0;
}

auto THVariable::requiresGrad() const -> bool {
  return requires_grad;
}

auto THVariable::isStochastic() const -> bool {
  return false;
}

auto THVariable::save() const -> SavedVariable {
  std::unique_ptr<Tensor> d(data->clone_shallow());
  auto expected_version = **version_counter;
  std::unique_ptr<THPVariableVersion> ref(version_counter->new_saved_ref());
  return SavedVariable(std::move(d), expected_version, std::move(ref));
}

auto SavedVariable::unpack() -> std::unique_ptr<thpp::Tensor>& {
  if (data) {
    int current_version = **version;
    if (expected_version != current_version) {
      throw std::runtime_error("one of the variables "
          "needed for gradient computation has been modified by an "
          "inplace operation");
    }
  }
  return data;
}
