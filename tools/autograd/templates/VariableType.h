#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

namespace at {

struct VariableType : public Type {
  VariableType(Context* context, Type* baseType);
  virtual ScalarType scalarType() override;
  virtual Backend backend() override;
  virtual bool isCuda() override;
  virtual bool isSparse() override;
  virtual bool isDistributed() override;
  virtual std::unique_ptr<Storage> storage() override;
  virtual std::unique_ptr<Storage> storage(size_t size) override;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size) override;
  virtual std::unique_ptr<Generator> generator() override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
  virtual size_t elementSizeInBytes() const override;
  static const char * typeString();
  Tensor unsafeTensorFromTH(void * th_pointer, bool retain) override;

  virtual void copy(const Tensor & src, Tensor & dst) override;
  ${type_derived_method_declarations}

private:
  Tensor & checked_unpack(const Tensor & t, const char * name, int pos) const;

private:
  Type* baseType;
};

} // namespace at
