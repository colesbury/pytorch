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
#include <stdarg.h>

extern THCState* state;

namespace {

void checkTypes(bool isCuda, thpp::Type type, ...) {
  va_list args;
  va_start(args, type);

  const char* name;
  while ((name = va_arg(args, const char*))) {
    thpp::Tensor* tensor = va_arg(args, thpp::Tensor*);
    printf("%s %p\n", name, tensor);
  }
}

}
