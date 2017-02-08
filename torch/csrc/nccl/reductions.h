#pragma once

#include "torch/lib/nccl/src/nccl.h"
#include <THPP/THPP.h>
#include <vector>

struct THCStream;

namespace torch { namespace nccl {

using comm_list_t = std::vector<ncclComm_t>;
using tensor_list = std::vector<std::unique_ptr<thpp::Tensor>>;

THCStream* get_stream(int device);

comm_list_t& communicator(const std::vector<int>& devices);

void synchronize_before(const std::vector<int>& devices);

std::vector<tensor_list> broadcast(
    const comm_list_t& comms,
    const tensor_list& inputs,
    const std::vector<int>& devices);


}}  // namespace torch::nccl
