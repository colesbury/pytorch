#include "DynamicTypes.h"

#include "THP.h"
#include <vector>
#include <unordered_map>


namespace torch {

// ordering must match enum DataType
static const char* TYPE_NAMES[] = {
  "Float",
  "Double",
  "Half",
  "Byte",
  "Char",
  "Short",
  "Int",
  "Long",
  nullptr,
};

static std::vector<DynamicType> types;
static std::unordered_map<PyObject *, size_t> class_map;

size_t getTypeIdx(const char *name, bool is_cuda, bool is_sparse)
{
  size_t type_idx = 0;
  while (1) {
    if (!TYPE_NAMES[type_idx]) {
      throw std::runtime_error("invalid type name");
    }
    if (strcmp(TYPE_NAMES[type_idx], name) == 0) {
      if (is_cuda) {
        type_idx |= FLAG_CUDA;
      }
      if (is_sparse) {
        type_idx |= FLAG_SPARSE;
      }
      return type_idx;
    }
    ++type_idx;
  }
}

size_t getTypeIdxForClass(PyObject *classobj)
{
  return class_map.at(classobj);
}

void registerType(size_t type_idx, const DynamicType& type)
{
  if (types.size() <= (size_t)type_idx) {
    types.resize(type_idx + 1);
  }
  types[type_idx] = type;
  class_map[type.classobj] = type_idx;
}

PyObject* getTHPTensorClass(size_t data_type)
{
  return types.at(data_type).classobj;
}

void THVoidTensor_retain(size_t data_type, THVoidTensor *cdata)
{
  types.at(data_type).retain(cdata);
}

void THVoidTensor_free(size_t data_type, THVoidTensor *cdata)
{
  types.at(data_type).free(cdata);
}

THVoidTensor* THVoidTensor_newWithSize(size_t data_type, THLongStorage *size)
{
  return (THVoidTensor*)types.at(data_type).newWithSize(size);
}

THLongStorage* THVoidTensor_newSizeOf(size_t data_type, THVoidTensor *size)
{
  return types.at(data_type).newSizeOf(size);
}


PyObject* THPVoidTensor_New(size_t data_type, THVoidTensor *cdata)
{
  PyTypeObject *type = (PyTypeObject *)types.at(data_type).classobj;
  PyObject *obj = type->tp_alloc(type, 0);
  // TODO: relase cdata on error
  if (obj) {
    ((THPVoidTensor *)obj)->cdata = cdata;
  }
  return obj;
}

}  // namespace
