#pragma once

#include <ATen/ATen.h>
#include <ATen/SmallVector.h>
#include <ATen/optional.h>

namespace at {

struct OperandInfo {
  OperandInfo() {}
  OperandInfo(const Tensor& t) : tensor_(t) {}

  /// Stride after broadcasting. The stride is in bytes, not number of elements.
  DimVector stride_;

  /// The original tensor operand. Note that the strides, data pointer, and
  /// other attributes may differ from due to dimension reordering and
  /// coalescing.
  Tensor tensor_;

  /// The desired type for the operand. This may be different from the actual
  /// tensor type, in which case casting is necessary.
  Type* type_ = nullptr;

  /// The data pointer. This may be different from tensor.data_ptr() if the
  /// iterator is split.
  void* data_ = nullptr;

  /// True if the kernel needs to handle a cast operation for this operand.
  bool needs_cast_ = false;
};

enum class IteratorFlags {
  COMMON_DTYPE = 1,
  ALLOW_CPU_SCALARS = 2,
};

struct SplitUntil32Bit;

struct TensorIterator {
  struct Builder;
  friend struct Builder;

  TensorIterator() {}

  using loop_t = const std::function<void(int, char**, const int64_t*, int64_t)>&;
  // using loop_t = void(*)(int, char**, int64_t*, int64_t);

  static TensorIterator binary_op(const Tensor& a, const Tensor& b, at::optional<Tensor> out=nullopt);
  static TensorIterator reduce_op(const Tensor& a, IntList dims);

  int ndim() const { return shape_.size(); }
  IntList shape() const { return shape_; }
  int64_t numel() const;
  int ntensors() const { return operands_.size(); }

  /// 1-dimensional iteration and no buffering or type conversion
  bool is_trivial_1d() const;

  IntList strides(int arg) const { return operands_[arg].stride_; }
  void* data_ptr(int arg) const;
  const Type& type(int arg=0) const {
    AT_ASSERT(operands_[arg].type_);
    return *operands_[arg].type_;
  }
  ScalarType dtype(int arg) { return type(arg).scalarType(); }
  bool is_scalar(int arg) const;
  bool is_cpu_scalar(int arg) const;

  bool needs_cast() const { return false; }

  void remove_operand(int arg);
  void remove_dimension(int dim);
  void narrow(int dim, int64_t start, int64_t size);

  Backend backend() const {
    return type().backend();
  }
  Tensor output(int arg=0) {
    return operands_[arg].tensor_;
  }

  std::unique_ptr<TensorIterator> split();

  template <typename T>
  T scalar_value(int arg) {
    if (isIntegralType(operands_[arg].tensor_.type().scalarType())) {
      return (T)integer_scalar_value(arg);
    } else {
      return (T)double_scalar_value(arg);
    }
  }

  int64_t integer_scalar_value(int arg);
  double double_scalar_value(int arg);

  void for_each(loop_t loop);
  void serial_for_each(loop_t loop, ArrayRef<char*> base_ptrs, IntList inner_strides, int64_t start, int64_t size);

  DimVector compatible_stride(int element_size) const;
  DimVector invert_perm(IntList input) const;

  DimVector make_counter(int64_t linear_offset) const;
  void increment_counter(DimVector& counter, int64_t n) const;
  DimVector get_inner_strides() const;
  SmallVector<char*, 4> get_data_ptrs(ArrayRef<char*> base, IntList counter) const;
  SmallVector<char*, 4> get_base_ptrs() const;

  bool can_use_32bit_indexing() const;
  SplitUntil32Bit with_32bit_indexing() const;

  // private
  void compute_shape();
  void compute_strides();
  void reorder_strides();
  void compute_common_type();
  void allocate_outputs();
  void coalesce_dimensions();
  void check_type_conversions();

  DimVector shape_;
  DimVector perm_;
  SmallVector<OperandInfo, 4> operands_;
  int num_outputs_ = 0;
};

struct TensorIterator::Builder {
  Builder() : iter_(new TensorIterator()) {};

  void add_output(optional<Tensor> output=nullopt) {
    iter_->operands_.emplace_back(output.value_or(Tensor()));
    iter_->num_outputs_++;
  }

  void add_input(const Tensor& input) {
    iter_->operands_.emplace_back(input);
  }

  std::unique_ptr<TensorIterator> build();

private:
  std::unique_ptr<TensorIterator> iter_;
};

/// A container-like struct that acts as if it contains splits of a
/// TensorIterator that can use 32-bit indexing. Taken together the splits cover
/// the original TensorIterator.
struct SplitUntil32Bit {
  struct iterator {
    iterator() {};
    iterator(const TensorIterator& iter);

    TensorIterator& operator*() const;
    iterator& operator++();
    bool operator!=(const iterator& other) {
      return !vec.empty() || !other.vec.empty();
    }

    /// stack of  queue
    std::vector<std::unique_ptr<TensorIterator>> vec;
  };

  SplitUntil32Bit(const TensorIterator& iter) : iter(iter) {}

  iterator begin() const;
  iterator end() const;

private:
  const TensorIterator& iter;
};

}  // namespace at
