#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <cudnn.h>
#include <THPP/THPP.h>

namespace torch { namespace cudnn {

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);
cudnnDataType_t getCudnnDataType(const thpp::Tensor& tensor);

}}  // namespace torch::cudnn

#endif
