#include "grad_buffer.h"

#include <THPP/THPP.h>
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

GradBuffer::GradBuffer(size_t size)
  : buffer(size)
  {}

auto GradBuffer::addGrad(size_t pos, std::shared_ptr<THVariable>&& var) -> void {
  auto& item = buffer[pos];
  if (!var) {
    return;
  }
  auto& tensor = var->data;
  if (!item.first) {
    buffer[pos] = std::make_pair<>(std::move(tensor), true);
  } else {
    if (item.second) {
      item.first.reset(item.first->clone());
      item.second = false;
    }
    item.first->cadd(*item.first, *tensor);
  }
}

auto GradBuffer::variables(GradBuffer&& g) -> std::vector<std::shared_ptr<THVariable>> {
  auto buffer = std::move(g.buffer);
  int size = buffer.size();
  std::vector<std::shared_ptr<THVariable>> result(size);
  for (int i = 0; i != size; ++i) {
    if (buffer[i].first) {
      result[i] = std::make_shared<THVariable>(
          std::move(buffer[i].first), 0, 1);
    }
  }
  return result;
}

}}  // namespace torch::autograd
