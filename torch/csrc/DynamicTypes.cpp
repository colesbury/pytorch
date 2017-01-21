#include "DynamicTypes.h"

#include "THP.h"
#include <vector>
#include <unordered_map>

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

PyTypeObject* getPyTypeObject(TensorType type)
{
  return tensortype_to_pytype.at(type);
}

}  // namespace
