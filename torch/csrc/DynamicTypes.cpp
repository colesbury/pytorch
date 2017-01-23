#include "DynamicTypes.h"

#include "THP.h"
#include <vector>
#include <unordered_map>
#include <THC/THC.h>
#include <THPP/tensors/THTensor.hpp>
#include <THPP/tensors/THCTensor.hpp>

extern THCState* state;

using namespace thpp;

namespace torch {

struct TensorTypeHasher
{
  std::size_t operator()(const TensorType& k) const
  {
    size_t hash = static_cast<size_t>(k.data_type);
    hash = (hash << 8) + k.is_cuda;
    hash = (hash << 1) + k.is_sparse;
    return hash;
  }
};


static std::unordered_map<std::string, Type> type_names = {
  {"Float", Type::FLOAT},
  {"Double", Type::DOUBLE},
  {"Half", Type::HALF},
  {"Byte", Type::UCHAR},
  {"Char", Type::CHAR},
  {"Short", Type::SHORT},
  {"Int", Type::INT},
  {"Long", Type::LONG},
};
static std::unordered_map<PyTypeObject*, TensorType> pytype_to_tensortype;
static std::unordered_map<TensorType, PyTypeObject*, TensorTypeHasher> tensortype_to_pytype;

TensorType getTensorType(const std::string& name, bool is_cuda, bool is_sparse)
{
  TensorType t;
  t.data_type = type_names.at(name);
  t.is_cuda = is_cuda;
  t.is_sparse = is_sparse;
  return t;
}

void registerType(TensorType type, PyTypeObject *pytype)
{
  pytype_to_tensortype[pytype] = type;
  tensortype_to_pytype[type] = pytype;
}

TensorType getTensorType(PyTypeObject *type)
{
  return pytype_to_tensortype.at(type);
}

PyTypeObject* getPyTypeObject(const thpp::Tensor& tensor)
{
  TensorType type;
  type.data_type = tensor.type();
  type.is_cuda = tensor.isCuda();
  type.is_sparse = false;

  return tensortype_to_pytype.at(type);
}

static std::unique_ptr<Tensor> createTensor(THVoidTensor *tensor, Type type, bool is_cuda)
{
  if (is_cuda) {
    if (type == Type::UCHAR) {
      return std::unique_ptr<Tensor>(new THCTensor<unsigned char>(state, (THCudaByteTensor*)tensor));
    } else if (type == Type::CHAR) {
      return std::unique_ptr<Tensor>(new THCTensor<char>(state, (THCudaCharTensor*)tensor));
    } else if (type == Type::SHORT) {
      return std::unique_ptr<Tensor>(new THCTensor<short>(state, (THCudaShortTensor*)tensor));
    } else if (type == Type::INT) {
      return std::unique_ptr<Tensor>(new THCTensor<int>(state, (THCudaIntTensor*)tensor));
    } else if (type == Type::LONG) {
      return std::unique_ptr<Tensor>(new THCTensor<long>(state, (THCudaLongTensor*)tensor));
    } else if (type == Type::FLOAT) {
      return std::unique_ptr<Tensor>(new THCTensor<float>(state, (THCudaTensor*)tensor));
    } else if (type == Type::DOUBLE) {
      return std::unique_ptr<Tensor>(new THCTensor<double>(state, (THCudaDoubleTensor*)tensor));
    } else if (type == Type::HALF) {
      return std::unique_ptr<Tensor>(new THCTensor<half>(state, (THCudaHalfTensor*)tensor));
    }
  } else if (type == Type::UCHAR) {
    return std::unique_ptr<Tensor>(new THTensor<unsigned char>((THByteTensor*)tensor));
  } else if (type == Type::CHAR) {
    return std::unique_ptr<Tensor>(new THTensor<char>((THCharTensor*)tensor));
  } else if (type == Type::SHORT) {
    return std::unique_ptr<Tensor>(new THTensor<short>((THShortTensor*)tensor));
  } else if (type == Type::INT) {
    return std::unique_ptr<Tensor>(new THTensor<int>((THIntTensor*)tensor));
  } else if (type == Type::LONG) {
    return std::unique_ptr<Tensor>(new THTensor<long>((THLongTensor*)tensor));
  } else if (type == Type::FLOAT) {
    return std::unique_ptr<Tensor>(new THTensor<float>((THFloatTensor*)tensor));
  } else if (type == Type::DOUBLE) {
    return std::unique_ptr<Tensor>(new THTensor<double>((THDoubleTensor*)tensor));
  }
  throw std::invalid_argument("passed character doesn't represent a tensor type");
}

std::unique_ptr<Tensor> createTensor(PyObject *data)
{
  auto tensor_type = getTensorType(Py_TYPE(data));
  auto type = tensor_type.data_type;
  auto tensor = ((THPVoidTensor *)data)->cdata;
  auto wrapper = createTensor(tensor, type, tensor_type.is_cuda);
  wrapper->retain();
  return wrapper;
}

PyObject* createPyObject(const thpp::Tensor& tensor)
{
  auto type = getPyTypeObject(tensor);
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    ((THPVoidTensor*)obj)->cdata = (THVoidTensor *)const_cast<thpp::Tensor&>(tensor).retain().cdata();
  }
  return obj;
}

}  // namespace
