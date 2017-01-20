#include "DynamicTypes.h"

#include "THP.h"
#include <vector>
#include <unordered_map>

using namespace thpp;

namespace torch {

static std::unordered_map<std::string, thpp::Type> type_names = {
  {"Float", Type::FLOAT},
  {"Double", Type::DOUBLE},
  {"Half", Type::HALF},
  {"Byte", Type::UCHAR},
  {"Char", Type::CHAR},
  {"Short", Type::SHORT},
  {"Int", Type::INT},
  {"Long", Type::LONG},
};
static std::unordered_map<PyTypeObject*, thpp::TensorType> pytype_to_tensortype;
static std::unordered_map<thpp::TensorType, PyTypeObject*> tensortype_to_pytype;

TensorType getTensorType(const std::string& name, bool is_cuda, bool is_sparse)
{
  TensorType t;
  t.data_type = type_names.at(name);
  t.is_cuda = is_cuda;
  t.is_sparse = is_sparse;
  return t;
}

void registerType(thpp::TensorType type, PyTypeObject *pytype)
{
  pytype_to_tensortype[pytype] = type;
  tensortype_to_pytype[type] = pytype;
}

thpp::TensorType getTensorType(PyTypeObject *type)
{
  return pytype_to_tensortype.at(type);
}

PyTypeObject* getPyTypeObject(thpp::TensorType type)
{
  return tensortype_to_pytype.at(type);
}

}  // namespace
