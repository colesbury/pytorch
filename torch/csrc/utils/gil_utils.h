#pragma once

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns the PyThreadState that last held (or currently holds) the GIL.
// Best-effort and racy: the result may be stale by the time the caller
// reads it. Returns NULL if the thread state is unavailable.
PyThreadState* torch_gil_last_holder(void);

#ifdef __cplusplus
}
#endif
