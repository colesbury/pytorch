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

int64_t VariableType::m_storage_offset(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_storage_offset(self_);
}
Tensor & VariableType::m_resize_(Tensor & self, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::zeros_out(IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::zeros(IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ones_out(IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ones(IntList size) {
    throw std::runtime_error("NYI");
}
int64_t VariableType::numel(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->numel(self_);
}
Tensor & VariableType::m_set_(Tensor & self, Storage & storage) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_set_(Tensor & self, const Tensor & source) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_set_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_fill_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
bool VariableType::m_is_same_size(const Tensor & self, const Tensor & other) {
    auto& self_ = checked_unpack(self, "self", 0);
    auto& other_ = checked_unpack(other, "other", 1);
    return baseType->m_is_same_size(self_, other_);
}
bool VariableType::m_is_contiguous(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_is_contiguous(self_);
}
bool VariableType::m_is_set_to(const Tensor & self, const Tensor & tensor) {
    auto& self_ = checked_unpack(self, "self", 0);
    auto& tensor_ = checked_unpack(tensor, "tensor", 1);
    return baseType->m_is_set_to(self_, tensor_);
}
Tensor & VariableType::m_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::masked_select(const Tensor & self, const Tensor & mask) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::t(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_t_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::squeeze_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::squeeze(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::squeeze_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::squeeze(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_squeeze_(Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_squeeze_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::unsqueeze_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::unsqueeze(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_unsqueeze_(Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::nonzero_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::nonzero(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_contiguous(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_clone(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_view(const Tensor & self, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_expand(const Tensor & self, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_resize_as_(Tensor & self, const Tensor & the_template) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::index_select_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::index_select(const Tensor & self, int64_t dim, const Tensor & index) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_narrow(const Tensor & self, int64_t dimension, int64_t start, int64_t length) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::range_out(Scalar start, Scalar end, Scalar step, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::range(Scalar start, Scalar end, Scalar step) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::range_out(Scalar start, Scalar end, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::range(Scalar start, Scalar end) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::gather_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::gather(const Tensor & self, int64_t dim, const Tensor & index) {
    throw std::runtime_error("NYI");
}
void* VariableType::m_data_ptr(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_data_ptr(self_);
}
bool VariableType::equal(const Tensor & self, const Tensor & other) {
    auto& self_ = checked_unpack(self, "self", 0);
    auto& other_ = checked_unpack(other, "other", 1);
    return baseType->equal(self_, other_);
}
Tensor & VariableType::__and___out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__and__(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__and___out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__and__(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__iand__(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__iand__(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__or___out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__or__(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__or___out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__or__(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ior__(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ior__(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__xor___out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__xor__(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__xor___out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__xor__(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ixor__(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ixor__(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__lshift___out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__lshift__(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__lshift___out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__lshift__(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ilshift__(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__ilshift__(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__rshift___out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__rshift__(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__rshift___out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::__rshift__(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__irshift__(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::__irshift__(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_lt(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_lt(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_lt_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_lt_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::lt_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::lt(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::lt_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::lt(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_gt(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_gt(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_gt_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_gt_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::gt_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::gt(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::gt_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::gt(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_le(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_le(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_le_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_le_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::le_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::le(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::le_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::le(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_ge(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_ge(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_ge_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_ge_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ge_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ge(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ge_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ge(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_eq(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_eq(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_eq_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_eq_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::eq_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::eq(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::eq_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::eq(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_ne(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::m_ne(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_ne_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_ne_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ne_out(const Tensor & tensor, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ne(const Tensor & tensor, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ne_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ne(const Tensor & tensor, const Tensor & other) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::min_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::min(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::min_out(const Tensor & self, int64_t dim, Tensor & min, Tensor & min_indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::min(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::min_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::min(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::min(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::max(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::max_out(const Tensor & self, int64_t dim, Tensor & max, Tensor & max_indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::max(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::max_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::max(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::max(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(const Tensor & self, int64_t k, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::kthvalue(const Tensor & self, int64_t k, int64_t dim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::mode(const Tensor & self, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(const Tensor & self, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::mode(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::mode(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::mode_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::mode(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::median_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, bool keepdim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::median_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::median_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::median(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(const Tensor & self, bool descending, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::sort(const Tensor & self, bool descending) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::sort(const Tensor & self, int64_t dim, bool descending) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(const Tensor & self, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::sort(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::sort_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::sort(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::topk(const Tensor & self, int64_t k) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::topk(const Tensor & self, int64_t k, int64_t dim, bool largest) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::topk_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::topk(const Tensor & self, int64_t k, int64_t dim) {
    throw std::runtime_error("NYI");
}
bool VariableType::m_all(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_all(self_);
}
bool VariableType::m_any(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_any(self_);
}
int64_t VariableType::m_get_device(const Tensor & self) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_get_device(self_);
}
Tensor & VariableType::abs_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::abs(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_abs_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sigmoid_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sigmoid_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sigmoid(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::log_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::log(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log1p_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::log1p_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::log1p(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::lgamma_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::lgamma(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_lgamma_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_exp_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::exp_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::exp(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cos_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cos_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cos(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_acos_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::acos_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::acos(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cosh_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cosh_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cosh(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sin_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sin_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sin(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_asin_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::asin_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::asin(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sinh_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sinh_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sinh(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_tan_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::tan_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tan(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_atan_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::atan_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::atan(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_tanh_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::tanh_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tanh(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sqrt_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sqrt_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sqrt(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_rsqrt_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::rsqrt_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::rsqrt(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_ceil_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ceil_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ceil(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_floor_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::floor_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::floor(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_round_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::round_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::round(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_trunc_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::trunc_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::trunc(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_frac_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::frac_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::frac(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mean_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mean(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mean_out(const Tensor & self, int64_t dim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mean(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::mean(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::var_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::var_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::var(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::var_out(const Tensor & self, int64_t dim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::var(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::var(const Tensor & self, bool unbiased) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::var(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::std_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::std_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::std(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::std_out(const Tensor & self, int64_t dim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::std(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::std(const Tensor & self, bool unbiased) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::std(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::norm_out(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::norm_out(const Tensor & self, Scalar p, int64_t dim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::norm(const Tensor & self, Scalar p, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::norm(const Tensor & self, Scalar p) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::norm(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::renorm_out(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::dist(const Tensor & self, const Tensor & other, Scalar p) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::dist(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::reciprocal_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::reciprocal(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_reciprocal_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::neg_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::neg(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_neg_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::atan2_out(const Tensor & self, const Tensor & other, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::atan2(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_atan2_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::pow_out(const Tensor & self, Scalar exponent, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::pow(const Tensor & self, Scalar exponent) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::pow(const Tensor & self, const Tensor & exponent) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_pow_(Tensor & self, Scalar exponent) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_pow_(Tensor & self, const Tensor & exponent) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::lerp(const Tensor & self, const Tensor & end, Scalar weight) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_lerp_(Tensor & self, const Tensor & end, Scalar weight) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::linspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::linspace(Scalar start, Scalar end, int64_t steps) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::linspace_out(Scalar start, Scalar end, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::linspace(Scalar start, Scalar end) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::logspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::logspace(Scalar start, Scalar end, int64_t steps) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::logspace_out(Scalar start, Scalar end, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::logspace(Scalar start, Scalar end) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::histc_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::histc(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::histc_out(const Tensor & self, int64_t bins, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::histc(const Tensor & self, int64_t bins) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::histc_out(const Tensor & self, int64_t bins, Scalar min, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::histc(const Tensor & self, int64_t bins, Scalar min) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::histc_out(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_zero_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sum_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sum(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sum_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sum(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::sum(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::prod_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::prod(const Tensor & self, int64_t dim, bool keepdim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::prod_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::prod(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::prod(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cumsum_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cumsum(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cumprod_out(const Tensor & self, int64_t dim, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cumprod(const Tensor & self, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sign_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sign(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sign_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::trace(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::add(const Tensor & self, Scalar value, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::add_out(const Tensor & self, Scalar value, SparseTensor other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::add(const Tensor & self, Scalar value, SparseTensor other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::add_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::add(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::add_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::add(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::add_out(const Tensor & self, SparseTensor other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::add(const Tensor & self, SparseTensor other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_add_(Tensor & self, Scalar value, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_add_(Tensor & self, Scalar value, SparseTensor other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_add_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_add_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_add_(Tensor & self, SparseTensor other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sub(const Tensor & self, Scalar value, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sub_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sub(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::sub_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::sub(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sub_(Tensor & self, Scalar value, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sub_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_sub_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mul_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mul(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mul_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mul(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_mul_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_mul_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::div_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::div(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::div_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::div(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_div_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_div_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::fmod_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::fmod(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::fmod_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::fmod(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_fmod_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_fmod_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::remainder_out(const Tensor & self, Scalar value, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::remainder(const Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::remainder_out(const Tensor & self, const Tensor & other, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::remainder(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_remainder_(Tensor & self, Scalar value) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_remainder_(Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::clamp_out(const Tensor & self, Scalar min, Scalar max, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::clamp(const Tensor & self, Scalar min, Scalar max) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::clamp_out(const Tensor & self, Scalar min, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::clamp(const Tensor & self, Scalar min) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_clamp_(Tensor & self, Scalar min, Scalar max) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_clamp_(Tensor & self, Scalar min) {
    throw std::runtime_error("NYI");
}
Scalar VariableType::dot(const Tensor & self, const Tensor & tensor) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::tril_out(const Tensor & self, int64_t diagonal, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tril(const Tensor & self, int64_t diagonal) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::tril_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tril(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_tril_(Tensor & self, int64_t diagonal) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_tril_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::triu_out(const Tensor & self, int64_t diagonal, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::triu(const Tensor & self, int64_t diagonal) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::triu_out(const Tensor & self, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::triu(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_triu_(Tensor & self, int64_t diagonal) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_triu_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cross_out(const Tensor & self, const Tensor & other, int64_t dim, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cross(const Tensor & self, const Tensor & other, int64_t dim) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cross_out(const Tensor & self, const Tensor & other, Tensor & destination) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cross(const Tensor & self, const Tensor & other) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::eye_out(int64_t n, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::eye(int64_t n) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::eye_out(int64_t n, int64_t m, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::eye(int64_t n, int64_t m) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::diag_out(const Tensor & self, int64_t diagonal, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::diag(const Tensor & self, int64_t diagonal) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::diag_out(const Tensor & self, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::diag(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmm_out(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmm_(Tensor & self, Scalar beta, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmv_out(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addmv_out(const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmv_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmv_(Tensor & self, Scalar beta, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addr_out(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addr_out(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addr_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addr_(Tensor & self, Scalar beta, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::ger_out(const Tensor & self, const Tensor & vec2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::ger(const Tensor & self, const Tensor & vec2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mv_out(const Tensor & self, const Tensor & vec, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mv(const Tensor & self, const Tensor & vec) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::mm_out(const Tensor & self, const Tensor & mat2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::mm(const Tensor & self, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::bmm_out(const Tensor & self, const Tensor & mat2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::bmm(const Tensor & self, const Tensor & mat2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::baddbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::baddbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_baddbmm_(Tensor & self, Scalar beta, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_baddbmm_(Tensor & self, Scalar beta, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addcmul_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addcdiv_(Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::gesv_out(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::gesv(const Tensor & self, const Tensor & A) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::gels(const Tensor & self, const Tensor & A) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(const Tensor & self, const Tensor & A, bool upper, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A, bool upper) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::trtrs_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::trtrs(const Tensor & self, const Tensor & A) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::symeig(const Tensor & self, bool eigenvectors, bool upper) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::symeig(const Tensor & self, bool eigenvectors) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::symeig_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::symeig(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::eig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::eig(const Tensor & self, bool eigenvectors) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::eig_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::eig(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::svd_out(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor,Tensor> VariableType::svd(const Tensor & self, bool some) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::svd_out(const Tensor & self, Tensor & res1, Tensor & res2, Tensor & res3) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor,Tensor> VariableType::svd(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::inverse_out(const Tensor & self, Tensor & output) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::inverse(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potrf_out(const Tensor & self, bool upper, Tensor & output) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potrf(const Tensor & self, bool upper) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potrf_out(const Tensor & self, Tensor & output) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potrf(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potrs_out(const Tensor & self, const Tensor & input2, bool upper, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potrs(const Tensor & self, const Tensor & input2, bool upper) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potrs_out(const Tensor & self, const Tensor & input2, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potrs(const Tensor & self, const Tensor & input2) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potri_out(const Tensor & self, bool upper, Tensor & output) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potri(const Tensor & self, bool upper) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::potri_out(const Tensor & self, Tensor & output) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::potri(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self, bool upper, Scalar tol) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(const Tensor & self, bool upper, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self, bool upper) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(const Tensor & self, Scalar tol, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self, Scalar tol) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::pstrf_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::pstrf(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::qr_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::qr(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::geqrf(const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,const Tensor &> VariableType::orgqr_out(const Tensor & self, const Tensor & input2, Tensor & result) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,const Tensor &> VariableType::orgqr(const Tensor & self, const Tensor & input2) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,const Tensor &> VariableType::ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & result) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,const Tensor &> VariableType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,const Tensor &> VariableType::ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, Tensor & result) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,const Tensor &> VariableType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,const Tensor &> VariableType::ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, Tensor & result) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,const Tensor &> VariableType::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::btrifact_out(const Tensor & info, bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::btrifact(const Tensor & info, bool pivot, const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::btrifact_out(const Tensor & info, const Tensor & self, Tensor & result, Tensor & pivots) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::btrifact(const Tensor & info, const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::btrifact_out(bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::btrifact(bool pivot, const Tensor & self) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor &,Tensor &> VariableType::btrifact_out(const Tensor & self, Tensor & result, Tensor & pivots) {
    throw std::runtime_error("NYI");
}
std::tuple<Tensor,Tensor> VariableType::btrifact(const Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::btrisolve_out(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::randperm_out(Generator & generator, int64_t n, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::randperm(Generator & generator, int64_t n) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::randperm_out(int64_t n, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::randperm(int64_t n) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::multinomial(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::multinomial(Generator & generator, const Tensor & self, int64_t num_samples) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::multinomial_out(const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::multinomial(const Tensor & self, int64_t num_samples, bool replacement) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::multinomial_out(const Tensor & self, int64_t num_samples, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::multinomial(const Tensor & self, int64_t num_samples) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self, Generator & generator, double from, double to) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self, Generator & generator, double from) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self, double from, double to) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self, Generator & generator) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self, double from) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_uniform_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self, Generator & generator, double median, double sigma) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self, Generator & generator, double median) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self, double median, double sigma) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self, Generator & generator) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self, double median) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_cauchy_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self, Generator & generator, double mean, double std) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self, Generator & generator, double mean) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self, double mean, double std) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self, Generator & generator) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self, double mean) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_log_normal_(Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::rand_out(Generator & generator, IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::rand(Generator & generator, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::rand_out(IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::rand(IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::randn_out(Generator & generator, IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::randn(Generator & generator, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::randn_out(IntList size, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::randn(IntList size) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_geometric_(Tensor & self, Generator & generator, double p) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_geometric_(Tensor & self, double p) {
    throw std::runtime_error("NYI");
}
int64_t VariableType::m_size(const Tensor & self, int64_t dim) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_size(self_, dim);
}
int64_t VariableType::m_stride(const Tensor & self, int64_t dim) {
    auto& self_ = checked_unpack(self, "self", 0);
    return baseType->m_stride(self_, dim);
}
Tensor VariableType::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tensor(Storage & storage, int64_t storageOffset, IntList size) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tensor(IntList size, IntList stride) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tensor(IntList size) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::tensor() {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::select_out(const Tensor & self, int dim, int64_t sliceIndex, Tensor & result) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::select(const Tensor & self, int dim, int64_t sliceIndex) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::m_assign_(Tensor & self, const Tensor & src) {
    throw std::runtime_error("NYI");
}
Tensor & VariableType::cat_out(TensorList tensors, int dim, Tensor & self) {
    throw std::runtime_error("NYI");
}
Tensor VariableType::cat(TensorList tensors, int dim) {
    throw std::runtime_error("NYI");
}
void VariableType::Abs_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Abs_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    throw std::runtime_error("NYI");
}
void VariableType::AbsCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::AbsCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights) {
    throw std::runtime_error("NYI");
}
void VariableType::BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights) {
    throw std::runtime_error("NYI");
}
void VariableType::BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    throw std::runtime_error("NYI");
}
void VariableType::ELU_updateOutput(const Tensor & input, const Tensor & output, Scalar alpha, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::ELU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar alpha, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::DistKLDivCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::DistKLDivCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::GatedLinear_updateOutput(const Tensor & input, const Tensor & output, int dim) {
    throw std::runtime_error("NYI");
}
void VariableType::GatedLinear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int dim) {
    throw std::runtime_error("NYI");
}
void VariableType::HardShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) {
    throw std::runtime_error("NYI");
}
void VariableType::HardShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) {
    throw std::runtime_error("NYI");
}
void VariableType::HardTanh_updateOutput(const Tensor & input, const Tensor & output, Scalar min_val, Scalar max_val, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::HardTanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar min_val, Scalar max_val, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::L1Cost_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::L1Cost_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    throw std::runtime_error("NYI");
}
void VariableType::L1Cost_updateGradInput(const Tensor & input, const Tensor & gradInput) {
    throw std::runtime_error("NYI");
}
void VariableType::LeakyReLU_updateOutput(const Tensor & input, const Tensor & output, Scalar negval, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::LeakyReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar negval, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    throw std::runtime_error("NYI");
}
void VariableType::GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    throw std::runtime_error("NYI");
}
void VariableType::GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    throw std::runtime_error("NYI");
}
void VariableType::GRUFused_updateGradInput(const Tensor & gradInInput, const Tensor & gradInHidden, const Tensor & gradOutput, const Tensor & gradInputHx, const Tensor & storage) {
    throw std::runtime_error("NYI");
}
void VariableType::LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    throw std::runtime_error("NYI");
}
void VariableType::LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    throw std::runtime_error("NYI");
}
void VariableType::LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    throw std::runtime_error("NYI");
}
void VariableType::LSTMFused_updateGradInput(const Tensor & storage, const Tensor & gradInGates, const Tensor & cx, const Tensor & cy, const Tensor & gradOutput, const Tensor & gradOutputCell, const Tensor & gradInputCx) {
    throw std::runtime_error("NYI");
}
void VariableType::LogSigmoid_updateOutput(const Tensor & input, const Tensor & output, const Tensor & buffer) {
    throw std::runtime_error("NYI");
}
void VariableType::LogSigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & buffer) {
    throw std::runtime_error("NYI");
}
void VariableType::LogSoftMax_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::LogSoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::MarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::MarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::MSECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::MSECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiLabelMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, const Tensor & isTarget, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiLabelMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, const Tensor & isTarget, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, const Tensor & weights, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, const Tensor & weights, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, Scalar margin) {
    throw std::runtime_error("NYI");
}
void VariableType::PReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, int64_t nOutputPlane) {
    throw std::runtime_error("NYI");
}
void VariableType::PReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int64_t nOutputPlane) {
    throw std::runtime_error("NYI");
}
void VariableType::PReLU_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradWeight, const Tensor & gradWeightBuf, const Tensor & gradWeightBuf2, int64_t nOutputPlane, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::Linear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & addBuffer) {
    throw std::runtime_error("NYI");
}
void VariableType::Linear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight) {
    throw std::runtime_error("NYI");
}
void VariableType::Linear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & addBuffer, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::RReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace, Generator & generator) {
    throw std::runtime_error("NYI");
}
void VariableType::RReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::Sigmoid_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Sigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Sigmoid_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::SmoothL1Criterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::SmoothL1Criterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftMax_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftPlus_updateOutput(const Tensor & input, const Tensor & output, Scalar beta, Scalar threshold) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftPlus_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar beta, Scalar threshold) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) {
    throw std::runtime_error("NYI");
}
void VariableType::SoftShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) {
    throw std::runtime_error("NYI");
}
void VariableType::IndexLinear_updateOutput(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & normalizedValues, int train) {
    throw std::runtime_error("NYI");
}
void VariableType::IndexLinear_accGradParameters(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, const Tensor & valuesBuffer, Scalar weightDecay, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SparseLinear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias) {
    throw std::runtime_error("NYI");
}
void VariableType::SparseLinear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, Scalar weightDecay, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::Sqrt_updateOutput(const Tensor & input, const Tensor & output, Scalar eps) {
    throw std::runtime_error("NYI");
}
void VariableType::Sqrt_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Square_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Square_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    throw std::runtime_error("NYI");
}
void VariableType::Tanh_updateOutput(const Tensor & input, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Tanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Tanh_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::Threshold_updateOutput(const Tensor & input, const Tensor & output, Scalar threshold, Scalar val, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::Threshold_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar threshold, Scalar val, bool inplace) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize, int outputFrameSize) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int dW) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int dW) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalRowConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalRowConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) {
    throw std::runtime_error("NYI");
}
void VariableType::TemporalRowConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDepthWiseConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionLocal_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionLocal_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialConvolutionLocal_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAdaptiveMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAdaptiveMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAdaptiveAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int owidth, int oheight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAdaptiveAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialFullConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int owidth, int oheight) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int kH, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int kH, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int kH, int dW, int dH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialUpSamplingBilinear_updateOutput(const Tensor & input, const Tensor & output, int outputHeight, int outputWidth) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialUpSamplingBilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialGridSamplerBilinear_updateOutput(const Tensor & input, const Tensor & grid, const Tensor & output) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialGridSamplerBilinear_updateGradInput(const Tensor & input, const Tensor & gradInput, const Tensor & grid, const Tensor & gradGrid, const Tensor & gradOutput) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kT, int kW, int kH, int dT, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kT, int kW, int kH, int dT, int dW, int dH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialReflectionPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialReflectionPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricUpSamplingTrilinear_updateOutput(const Tensor & input, const Tensor & output, int outputDepth, int outputHeight, int outputWidth) {
    throw std::runtime_error("NYI");
}
void VariableType::VolumetricUpSamplingTrilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialCrossMapLRN_updateOutput(const Tensor & input, const Tensor & output, const Tensor & scale, int size, Scalar alpha, Scalar beta, Scalar k) {
    throw std::runtime_error("NYI");
}
void VariableType::SpatialCrossMapLRN_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & scale, const Tensor & output, int size, Scalar alpha, Scalar beta, Scalar k) {
    throw std::runtime_error("NYI");
}


}
