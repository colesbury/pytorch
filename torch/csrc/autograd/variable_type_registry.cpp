#include "variable_type_registry.h"

#include "VariableType.h"

#include <mutex>
#include <unordered_map>

using namespace at;

namespace torch { namespace autograd {

Type* VariableTypeRegistry::get(const Tensor& tensor)
{
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return get(tensor.type());
}

Type* VariableTypeRegistry::get(const Type& baseType)
{
  static std::once_flag once;
  std::call_once(once, []() {
    auto context = &at::globalContext();

    std::vector<Backend> backends = {
      Backend::CPU,
      Backend::CUDA,
      Backend::SparseCPU,
      Backend::SparseCUDA
    };

    std::vector<ScalarType> scalarTypes = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half
    };

    for (auto backend : backends) {
      for (auto scalarType : scalarTypes) {
        if (scalarType == ScalarType::Half
            && (backend == Backend::SparseCPU || backend == Backend::SparseCUDA)) {
          continue;
        }

        auto baseType = &context->getType(backend, scalarType);
        types[baseType].reset(new VariableType(context, baseType));
      }
    }
  });
  return types[&baseType].get();
}

std::unordered_map<const Type*, std::unique_ptr<Type>> VariableTypeRegistry::types;


}} // namespace torch::autograd
