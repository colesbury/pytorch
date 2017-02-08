#include "torch/csrc/autograd/engine.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <THPP/THPP.h>
#include <thread>
#include <unordered_set>
#include <typeinfo>
#include <sstream>

#ifdef WITH_CUDA
#include <cuda.h>
#include <THC/THC.h>
#endif

using thpp::Tensor;

namespace torch { namespace autograd {

struct ReadyQueue {
  std::deque<FunctionTask> queue;
  std::condition_variable not_empty;
  std::mutex mutex;

  void push_front(FunctionTask item);
  FunctionTask pop_back();
};

struct BackwardTask {
  std::exception_ptr exception;
  std::atomic_bool has_error;
  std::atomic<uint64_t> outstanding_tasks;
  bool retain_variables;

  std::mutex mutex;
  std::condition_variable not_done;
  std::unordered_map<Function*, GradBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;

  BackwardTask(bool retain_variables)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , retain_variables(retain_variables)
    , mutex()
    , not_done()
    , not_ready()
    , dependencies() {}
};

struct FunctionTask {
  BackwardTask* base;
  std::shared_ptr<Function> fn;
  GradBuffer grad;

  FunctionTask(BackwardTask* base, std::shared_ptr<Function> fn, GradBuffer grad)
    : base(base)
    , fn(fn)
    , grad(std::move(grad)) {}
};

auto ReadyQueue::push_front(FunctionTask item) -> void {
  ++item.base->outstanding_tasks;
  {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_front(std::move(item));
  }
  not_empty.notify_one();
}

auto ReadyQueue::pop_back() -> FunctionTask {
  std::unique_lock<std::mutex> lock(mutex);
  not_empty.wait(lock, [this]{ return !queue.empty(); });
  auto value = std::move(queue.back()); queue.pop_back();
  return value;
}

Engine::Engine() : ready_queues() {
}

Engine::~Engine() = default;

auto Engine::thread_main(ReadyQueue& queue) -> void {
  while (1) {
    FunctionTask task = queue.pop_back();
    if (!task.base->has_error.load()) {
      try {
        evaluate_function(task);
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }
    if (--task.base->outstanding_tasks == 0) {
      task.base->not_done.notify_all();
    }
  }
}

auto Engine::thread_on_exception( FunctionTask& task, std::exception& e) -> void {
  std::lock_guard<std::mutex> lock(task.base->mutex);
  if (!task.base->has_error.load()) {
    task.base->exception = std::current_exception();
    task.base->has_error = true;
  }
}

auto Engine::evaluate_function(FunctionTask& task) -> void
{
  auto& fn = *task.fn;
  auto grad_output = fn.call_hooks(GradBuffer::variables(std::move(task.grad)));
  auto grad_inputs = fn.apply(grad_output);
  if (!task.base->retain_variables) {
    fn.releaseVariables();
  }

  if (grad_inputs.size() != fn.previous_functions.size()) {
    std::stringstream ss;
    ss << "Function '" << fn.name() << "' returned an invalid number of gradients - expected ";
    ss << fn.previous_functions.size() << ", but got " << grad_inputs.size();
    throw std::runtime_error(ss.str());
  }

  int size = grad_inputs.size();
  for (int i = 0; i < size; ++i) {
    auto& grad_input = grad_inputs[i];
    auto& prev_fn = fn.previous_functions[i].first;
    int output_nr = fn.previous_functions[i].second;

    // null inputs have no previous_function and we skip them here
    if (!prev_fn) {
      continue;
    }

    // Stochastic functions are placed in the ready queue by
    // compute_dependencies, so we can skip them here.
    if (prev_fn->is_stochastic || !prev_fn->requires_grad) {
      continue;
    }

    std::lock_guard<std::mutex> lock(task.base->mutex);
    if (auto var = dynamic_cast<Variable*>(prev_fn.get())) {
      var->backward(grad_input);
      continue;
    }

    // Check if the function is ready for backward
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;
    auto it = dependencies.find(prev_fn.get());
    auto name = prev_fn->name();
    if (it == dependencies.end()) {
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      dependencies.erase(it);
      is_ready = true;
    }

    auto& not_ready = task.base->not_ready;
    auto not_ready_it = not_ready.find(prev_fn.get());
    if (not_ready_it == not_ready.end()) {
      // No buffers have been allocated for the function
      GradBuffer prev_buffer(prev_fn->num_outputs);
      prev_buffer.addGrad(output_nr, std::move(grad_input));
      if (is_ready) {
        auto& queue = ready_queue(prev_buffer.device());
        queue.push_front(FunctionTask(task.base, prev_fn, std::move(prev_buffer)));
      } else {
        not_ready.emplace(prev_fn.get(), std::move(prev_buffer));
      }
    } else {
      // The function already has a buffer
      auto &prev_buffer = not_ready_it->second;
      prev_buffer.addGrad(output_nr, std::move(grad_input));
      if (is_ready) {
        auto& queue = ready_queue(prev_buffer.device());
        queue.push_front(FunctionTask(task.base, prev_fn, std::move(prev_buffer)));
        not_ready.erase(not_ready_it);
      }
    }
  }
}

auto Engine::compute_dependencies(function_queue queue, BackwardTask& task) -> void {
  // First, search the graph and find all stochastic functions. Append them to the queue.
  std::unordered_set<Function*> seen;
  function_queue search_queue(queue);
  while (search_queue.size() > 0) {
    auto fn = search_queue.back(); search_queue.pop_back();
    for (auto& prev_fn_pair : fn->previous_functions) {
      auto& prev_fn = prev_fn_pair.first;
      Function* prev_ptr = prev_fn.get();
      if (!prev_ptr) continue;
      if (prev_ptr->is_stochastic && prev_ptr->requires_grad && seen.count(prev_ptr) == 0) {
        ready_queue(-1).push_front(FunctionTask(&task, prev_fn, GradBuffer(0)));
        queue.push_back(prev_ptr);
      }
      if (seen.count(prev_ptr) == 0) {
        seen.insert(prev_ptr);
        search_queue.push_back(prev_ptr);
      }
    }
  }

  // Now, queue contains all nodes that will start propagating gradients. We no longer have
  // to expand functions that don't require grad.
  auto& dependencies = task.dependencies;
  seen.clear();
  // Just to make sure that they will never be added to the queue again
  seen.insert(queue.begin(), queue.end());
  while (queue.size() > 0) {
    auto fn = std::move(queue.back()); queue.pop_back();
    // This is needed only to filter out backward roots that don't require grad
    if (!fn->requires_grad) continue;
    for (auto& prev_fn_pair : fn->previous_functions) {
      Function* prev_ptr = prev_fn_pair.first.get();
      if (!prev_ptr) continue;
      if (dynamic_cast<Variable*>(prev_ptr)) continue;
      if (!prev_ptr->requires_grad) continue;
      if (prev_ptr->is_stochastic) continue; // Stochastic nodes were in the queue already
      dependencies[prev_ptr] += 1;
      if (seen.count(prev_ptr) == 0) {
        seen.insert(prev_ptr);
        queue.push_back(prev_ptr);
      }
    }
  }
}

auto Engine::backward(const variable_list& variables,
                      tensor_list& grad_variables,
                      bool retain_variables) -> void {
  static std::once_flag once_flag;
  std::call_once(once_flag, [this]{
    int num_devices = 0;
#ifdef WITH_CUDA
    THCudaCheck(cudaGetDeviceCount(&num_devices));
#endif
    ready_queues = std::vector<std::unique_ptr<ReadyQueue>>(num_devices + 1);
    for (auto& queue : ready_queues) {
      queue.reset(new ReadyQueue());
      std::thread t(&Engine::thread_main, this, std::ref(*queue));
      t.detach();
    }
  });

  BackwardTask backward_task(retain_variables);
  std::unique_lock<std::mutex> lock(backward_task.mutex);

  function_queue creators;
  // bool did_leaf_backward = false;
  int size = variables.size();
  for (int i = 0; i < size; ++i) {
    auto& var = variables[i];
    auto& grad = grad_variables[i];
    if (!var->creator) {
      // If someone calls .backward() on a leaf, it's simple...
      if (var->requires_grad) {
        var->backward(std::make_shared<Variable>(std::move(grad), false, true));
        // did_leaf_backward = true;
      }
    } else {
      creators.push_back(var->creator.get());
      if (var->creator->requires_grad) {
        GradBuffer buf(var->creator->num_outputs);
        buf.addGrad(var->output_nr, Variable::of(std::move(grad)));

        auto& queue = ready_queue(var->data->getDevice());
        queue.push_front(FunctionTask(&backward_task, var->creator, std::move(buf)));
      }
    }
  }

  compute_dependencies(std::move(creators), backward_task);

  backward_task.not_done.wait(lock, [&backward_task]{
    return backward_task.outstanding_tasks.load() == 0;
  });

  // std::exception_ptr eptr = backward_task.exception.load();
  if (backward_task.has_error.load()) {
    // throw std::runtime_error("has_error");
    // try {
    //   printf("throwing exception\n");
    //   std::rethrow_exception(eptr);
    // } catch (std::exception& e) {
    //     printf("catching exception\n");
    //    std::cout << "Caught exception \"" << e.what() << "\"\n";
    // }
    // printf("rethrowing exception\n");
    std::rethrow_exception(backward_task.exception);
  }

  if (!backward_task.not_ready.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }
}

auto Engine::ready_queue(int device) -> ReadyQueue& {
  return *ready_queues.at(device + 1);
}

// std::vector<ReadyQueue> Engine::ready_queues;

}} // namespace torch::autograd
