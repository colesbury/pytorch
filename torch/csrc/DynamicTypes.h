#ifndef THP_DYNAMIC_TYPES_INC
#define THP_DYNAMIC_TYPES_INC

#include "Types.h"
#include <Python.h>
#include <cstddef>
#include <functional>

struct THLongStorage;


namespace torch {

enum DataType {
  FLOAT = 0,
  DOUBLE = 1,
  HALF = 2,
  BYTE = 3,
  CHAR = 4,
  SHORT = 5,
  INT = 6,
  LONG = 7
};
const int FLAG_CUDA = 0x10;
const int FLAG_SPARSE = 0x20;

typedef std::function<void(void*)> FreeFn;
typedef std::function<void(void*)> RetainFn;
typedef std::function<void*(THLongStorage*)> NewWithSizeFn;
typedef std::function<THLongStorage*(void*)> NewSizeOfFn;

struct DynamicType {
  PyObject *classobj;
  FreeFn free;
  RetainFn retain;
  NewWithSizeFn newWithSize;
  NewSizeOfFn newSizeOf;
};

size_t getTypeIdx(const char *type_name, bool is_cuda, bool is_sparse);
size_t getTypeIdxForClass(PyObject *pyclass);
void registerType(size_t data_type, const DynamicType& type);
PyObject* getTHPTensorClass(size_t data_type);

void THVoidTensor_retain(size_t data_type, THVoidTensor *cdata);
void THVoidTensor_free(size_t data_type, THVoidTensor *cdata);
THVoidTensor* THVoidTensor_newWithSize(size_t data_type, THLongStorage *size);
THLongStorage* THVoidTensor_newSizeOf(size_t data_type, THVoidTensor *cdata);

PyObject * THPVoidTensor_New(size_t data_type, THVoidTensor *cdata);


}  // namespace torch

#endif
