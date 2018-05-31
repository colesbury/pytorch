#include "DispatchStub.h"

#include <cpuinfo.h>
#include <cstdlib>
#include <strings.h>

namespace at { namespace native {

static CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
    if (strcasecmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
    if (strcasecmp(envar, "avx") == 0) {
      return CPUCapability::AVX;
    }
    if (strcasecmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    std::cerr << "ignoring invalid value for ATEN_CPU_CAPABILITY: " << envar << "\n";
  }

#ifndef __powerpc__
  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
    if (cpuinfo_has_x86_avx()) {
      return CPUCapability::AVX;
    }
  }
#endif
  return CPUCapability::DEFAULT;
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

}}  // namespace at::native
