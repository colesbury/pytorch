#pragma once

#include <Python.h>

inline bool THPUtils_checkLong(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

inline long THPUtils_unpackLong(PyObject* obj) {
  if (PyLong_Check(obj)) {
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0) {
      throw std::runtime_error("Overflow when unpacking long");
    }
    return (long)value;
  }
#if PY_MAJOR_VERSION == 2
  if (PyInt_Check(obj)) {
    return PyInt_AS_LONG(obj);
  }
#endif
  throw std::runtime_error("Could not unpack long");
}

inline bool THPUtils_checkDouble(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj);
#else
  return PyFloat_Check(obj) || PyLong_Check(obj);
#endif
}

inline double THPUtils_unpackDouble(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  if (PyLong_Check(obj)) {
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0) {
      throw std::runtime_error("Overflow when unpacking double");
    }
    return (double)value;
  }
#if PY_MAJOR_VERSION == 2
  if (PyInt_Check(obj)) {
    return (double)PyInt_AS_LONG(obj);
  }
#endif
  throw std::runtime_error("Could not unpack double");
}
