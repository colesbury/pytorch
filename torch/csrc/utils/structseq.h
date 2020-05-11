#pragma once

#include "torch/csrc/python_headers.h"

namespace torch { namespace utils {

#if PY_MAJOR_VERSION == 2
PyObject *structseq_slice(PyStructSequence *obj, Py_ssize_t low, Py_ssize_t high);
#endif

PyTypeObject *init_struct_seq(PyTypeObject *tp, PyStructSequence_Desc *desc);

}}
