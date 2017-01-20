#ifndef THP_DYNAMIC_TYPES_INC
#define THP_DYNAMIC_TYPES_INC

#include "Types.h"
#include <Python.h>
#include <cstddef>
#include <functional>
#include <THPP/THPP.h>

struct THLongStorage;


namespace torch {

thpp::TensorType getTensorType(const std::string& name, bool is_cuda, bool is_sparse);
void registerType(thpp::TensorType type, PyTypeObject *pytype);
thpp::TensorType getTensorType(PyTypeObject *type);
PyTypeObject* getPyTypeObject(thpp::TensorType type);

}  // namespace torch

#endif
