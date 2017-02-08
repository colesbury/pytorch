#ifndef THC_STREAM_HPP
#define THC_STREAM_HPP

#include "THCStream.h"


#include <cuda_runtime_api.h>
#include "THCGeneral.h"

struct THCStream
{
    cudaStream_t stream;
    int device;
    int refcount;
};


THC_API THCStream* THCStream_new(int flags);
THC_API THCStream* THCStream_newWithPriority(int flags, int priority);
THC_API void THCStream_free(THCStream* self);
THC_API void THCStream_retain(THCStream* self);

#endif // THC_STREAM_HPP
