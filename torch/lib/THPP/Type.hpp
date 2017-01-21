#pragma once

namespace thpp {

/*
 * The following notation comes from:
 * docs.python.org/3.5/library/struct.html#module-struct
 * except from 'T', which stands for Tensor
 */

enum class Type : char {
  CHAR = 'c',
  UCHAR = 'B',
  FLOAT = 'f',
  DOUBLE = 'd',
  HALF = 'a',
  SHORT = 'h',
  USHORT = 'H',
  INT = 'i',
  UINT = 'I',
  LONG = 'l',
  ULONG = 'L',
  LONG_LONG = 'q',
  ULONG_LONG = 'Q',
  LONG_STORAGE = 'X',
  TENSOR = 'T',
  STORAGE = 'S',
};

struct TensorType {
  Type data_type;
  bool is_cuda;
  bool is_sparse;

  friend bool operator==(const TensorType &t1, const TensorType &t2)
  {
    return (t1.data_type == t2.data_type &&
            t1.is_cuda == t2.is_cuda &&
            t1.is_sparse == t2.is_sparse);
  }

  friend bool operator!=(const TensorType &t1, const TensorType &t2)
  {
    return !(t1 == t2);
  }
};

inline bool isFloat(Type t) {
  return (t == Type::FLOAT || t == Type::DOUBLE);
}

inline bool isObject(Type t) {
  return (t == Type::TENSOR || t == Type::STORAGE);
}

inline bool isInteger(Type t) {
  return (t == Type::CHAR || t == Type::UCHAR ||
          t == Type::SHORT || t == Type:: USHORT ||
          t == Type::INT || t == Type::UINT ||
          t == Type::LONG || t == Type::ULONG ||
          t == Type::LONG_LONG || t == Type::ULONG_LONG);
}

} // namespace thpp
