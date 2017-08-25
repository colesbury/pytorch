#include "VariableType.h"

#include "VariableTensor.h"
#include <iostream>

using torch::autograd::VariableTensor;

namespace at {

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context)
  , baseType(baseType) {
}

ScalarType VariableType::scalarType() {
  return baseType->scalarType();
}
Backend VariableType::backend() {
  return baseType->backend();
}
bool VariableType::isCuda() { return baseType->isCuda(); }
bool VariableType::isSparse() { return baseType->isSparse(); }
bool VariableType::isDistributed() { return baseType->isDistributed(); }

std::unique_ptr<Storage> VariableType::storage() {
  return baseType->storage();
}
std::unique_ptr<Storage> VariableType::storage(size_t size) {
  return baseType->storage(size);
}
std::unique_ptr<Storage> VariableType::storageFromBlob(void * data, int64_t size) {
  return baseType->storageFromBlob(data, size);
}
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) {
  return baseType->unsafeTensorFromTH(th_pointer, retain);
}
std::unique_ptr<Generator> VariableType::generator() {
  return baseType->generator();
}

const char * VariableType::toString() const {
  return VariableType::typeString();
}
TypeID VariableType::ID() const {
  throw std::runtime_error("VariableType::ID() not implemented");
}

const char * VariableType::typeString() {
  return "VariableType";
}

void VariableType::copy(const Tensor & src, Tensor & dst) {
  throw std::runtime_error("VariableType::copy() not implemented");
}

Tensor & VariableType::checked_unpack(const Tensor & t, const char * name, int pos) const
{
 if(!t.defined()) {
   runtime_error("Expected a Tensor of type %s but found an undefined Tensor for argument #%d '%s'",
     toString(),pos,name);
 }
 if (&t.type() == this) {
   return static_cast<VariableTensor*>(t.pImpl)->data;
 }
 runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
   toString(),t.type().toString(),pos,name);
}

${type_derived_method_definitions}

}
