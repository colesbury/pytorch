#define Py_BUILD_CORE
#include <Python.h>
#include <internal/pycore_gil.h>

#if PY_VERSION_HEX >= 0x030E0000
// 3.14+: interp struct definition split into its own header
#include <internal/pycore_interp_structs.h>
#else
// 3.12-3.13: interp struct lives here
#include <internal/pycore_interp.h>
#endif

#undef Py_BUILD_CORE

PyThreadState* torch_gil_last_holder(void) {
    PyThreadState* tstate = PyGILState_GetThisThreadState();
    if (!tstate || !tstate->interp)
        return NULL;
    struct _gil_runtime_state* gil = tstate->interp->ceval.gil;
    if (!gil)
        return NULL;

#if PY_VERSION_HEX >= 0x030E0000
    // 3.14+: last_holder is PyThreadState*
    return gil->last_holder;
#else
    // 3.12-3.13: last_holder is _Py_atomic_address
    return (PyThreadState*)_Py_atomic_load_relaxed(&gil->last_holder);
#endif
}
