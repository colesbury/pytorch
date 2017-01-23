#ifndef THP_AUTOGRAD_H
#define THP_AUTOGRAD_H

PyObject * THPAutograd_initExtension(PyObject *_unused);

#include "variable.h"
#include "function.h"
#include "engine.h"
#include "native_function.h"

#endif
