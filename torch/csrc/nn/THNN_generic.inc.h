#include "THNN_generic.h"

#include <TH/TH.h>
#include <THC/THC.h>
#include <THNN/THNN.h>
#ifdef THNN_
#undef THNN_
#endif
#include <THCUNN/THCUNN.h>
#ifdef THNN_
#undef THNN_
#endif
#include <sstream>
#include <stdarg.h>

extern THCState* state;

namespace {

static std::runtime_error invalid_tensor(const char* expected, const char* got) {
  std::stringstream ss;
  ss << "expected " << expected << " tensor (got " << got << " tensor)";
  return std::runtime_error(ss.str());
}

void checkTypes(bool isCuda, thpp::Type type, ...) {
  va_list args;
  va_start(args, type);

  const char* name;
  while ((name = va_arg(args, const char*))) {
    thpp::Tensor* tensor = va_arg(args, thpp::Tensor*);
    if (!tensor) {
      continue;
    }
    if (tensor->isCuda() != isCuda) {
      throw invalid_tensor(isCuda ? "CUDA" : "CPU", tensor->isCuda() ? "CUDA" : "CPU");
    }
  }
}

}
