#include "grad_buffer.h"

#include <THPP/THPP.h>

namespace torch { namespace autograd {

GradBuffer::GradBuffer(size_t size)
  : buffer(size)
  {}

void GradBuffer::addGrad(size_t pos, std::unique_ptr<thpp::Tensor> tensor) {
  auto& item = buffer[pos];
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

std::vector<std::unique_ptr<thpp::Tensor>> GradBuffer::tensors() {
  std::vector<std::unique_ptr<thpp::Tensor>> result(buffer.size());
  for (size_t i = 0; i != buffer.size(); ++i) {
    result[i] = std::move(buffer[i].first);
  }
  return result;
}

thpp::Tensor& GradBuffer::operator[](size_t pos) {
  return *buffer[pos].first;
}

const thpp::Tensor& GradBuffer::operator[](size_t pos) const {
  return *buffer[pos].first;
}


}}  // namespace torch::autograd
