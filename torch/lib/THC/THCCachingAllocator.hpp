#ifndef THC_DEVICE_ALLOCATOR_HPP_INC
#define THC_DEVICE_ALLOCATOR_HPP_INC

#include <mutex>

THC_API std::mutex* THCCachingAllocator_getCudaFreeMutex();

#endif
