#include <Python.h>
#include <structmember.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/cuda/AutoGPU.h"


using namespace torch;
using namespace thpp;


PyObject *THPVariableClass = NULL;

THVariable::THVariable(thpp::TensorType tensor_type, std::unique_ptr<thpp::Tensor> data, char requires_grad, char is_volatile)
    : refcount(1), tensor_type(tensor_type), data(std::move(data)),
      creator(nullptr), grad(nullptr),
      version_counter(new THPVariableVersion()), output_nr(0),
      is_volatile(is_volatile), requires_grad(requires_grad),
      backward_hooks(nullptr), pyobj(nullptr)
{
}

THVariable::~THVariable()
{
}

void THVariable::free()
{
  if (THAtomicDecrementRef(&refcount)) {
    delete this;
  }
}

void THVariable::retain()
{
  THAtomicIncrementRef(&refcount);
}

bool THVariable::is_cuda()
{
  return data->isCuda();
}


PyObject * THPVariable_Wrap(THVariable *var)
{
  if (var->pyobj) {
    Py_INCREF(var->pyobj);
  } else {
    PyTypeObject *type = (PyTypeObject *)THPVariableClass;
    var->pyobj = type->tp_alloc(type, 0);
    if (var->pyobj) {
      ((THPVariable *)var->pyobj)->cdata = var;
      var->retain();
    }
  }
  return var->pyobj;
}

PyObject * THPVariable_New2(PyTypeObject *type, PyObject *data, PyObject *creator, char requires_grad, char is_volatile)
{
  THPUtils_assert(THPModule_isTensor(data), "data must be a Tensor");
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    auto var = (THPVariable*) obj;
    auto tensor_type = getTensorType(Py_TYPE(data));
    var->cdata = new THVariable(tensor_type, createTensor(data), requires_grad, is_volatile);
    var->cdata->creator = creator;
    var->cdata->pyobj = obj;
    Py_XINCREF(creator);
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
  Py_VISIT(self->cdata->creator);
  Py_VISIT(self->cdata->backward_hooks);
  return 0;
}

static int THPVariable_clear(THPVariable *self)
{
  Py_CLEAR(self->cdata->creator);
  Py_CLEAR(self->cdata->backward_hooks);
  return 0;
}

static void THPVariable_dealloc(THPVariable* self)
{
  PyObject_GC_UnTrack(self);
  Py_XDECREF(self->cdata->creator);
  Py_XDECREF(self->cdata->backward_hooks);
  self->cdata->pyobj = nullptr;
  self->cdata->free();
  self->cdata = nullptr;
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
  return PyInt_FromLong(**self->cdata->version_counter);
}

PyObject *THPVariable_get_creator(THPVariable *self)
{
  PyObject *creator = self->cdata->creator;
  if (!creator) {
    Py_RETURN_NONE;
  }
  Py_INCREF(creator);
  return creator;
}

PyObject * THPVariable_get_data(THPVariable *self)
{
  auto& data = *self->cdata->data;
  PyTypeObject* type = getPyTypeObject(self->cdata->tensor_type);
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THPVoidTensor*)obj)->cdata = (THVoidTensor *)data.retain().cdata();
  }
  return obj;
}

int THPVariable_set_data(THPVariable *self, PyObject *data)
{
  THPUtils_assertRet(-1, THPModule_isTensor(data), "Variable data has to "
      "be a tensor, but got %s", THPUtils_typename(data));
  auto tensor = createTensor(data);
  self->cdata->data.swap(tensor);
  self->cdata->tensor_type = getTensorType(Py_TYPE(data));
  return 0;
}

PyObject *THPVariable_get_raw_grad(THPVariable *self)
{
  THVariable *var = self->cdata;
  if (!var->grad) {
    Py_RETURN_NONE;
  }
  return THPVariable_Wrap(var->grad);
}

int THPVariable_set_raw_grad(THPVariable *self, PyObject *data)
{
  THVariable* grad = nullptr;
  if (data != Py_None) {
    THPUtils_assertRet(-1, THPVariable_Check(data),
        "expected Variable or None (got %s)", THPUtils_typename(data));
    grad = ((THPVariable*)data)->cdata;
    grad->retain();
  }
  if (self->cdata->grad) {
    self->cdata->grad->free();
  }
  self->cdata->grad = grad;
  return 0;
}

PyObject *THPVariable_get_grad(THPVariable *self)
{
  THVariable *var = self->cdata;
  if (!var->grad) {
    THCPAutoGPU __guard(var->data->getDevice());
    auto grad = var->data->newTensor();
    grad->resizeAs(*var->data).zero();
    var->grad = new THVariable(var->tensor_type, std::move(grad), 0, 1);
  }
  return THPVariable_Wrap(var->grad);
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  return PyBool_FromLong(self->cdata->is_volatile);
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "volatile must be a bool");
  self->cdata->is_volatile = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_output_nr(THPVariable *self)
{
  return PyInt_FromLong(self->cdata->output_nr);
}

PyObject *THPVariable_get_requires_grad(THPVariable *self)
{
  return PyBool_FromLong(self->cdata->requires_grad);
}

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "requires_grad must be a bool");
  if (self->cdata->creator) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  self->cdata->requires_grad = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self)
{
  if (self->cdata->backward_hooks) {
    Py_INCREF(self->cdata->backward_hooks);
    return self->cdata->backward_hooks;
  }
  Py_RETURN_NONE;
}

int THPVariable_set_backwards_hooks(THPVariable *self, PyObject *obj)
{
  Py_INCREF(obj);
  Py_XDECREF(self->cdata->backward_hooks);
  self->cdata->backward_hooks = obj;
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
