#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/util/python_stub.h>
#include <optional>

#include <atomic>

namespace c10::impl {

struct C10_API PyObjectSlot {
 public:
  PyObjectSlot() : pyobj_interpreter_(nullptr), pyobj_(nullptr) {}

  // Associate the TensorImpl with the specified PyObject
  void init_pyobj(PyObject* pyobj) {
    pyobj_interpreter_.store(
        getGlobalPyInterpreter(), std::memory_order_release);
    pyobj_.store(pyobj, std::memory_order_release);
  }

  PyObject* init_once_atomic(PyInterpreter* interpreter, PyObject* pyobj) {
    PyObject* expected = nullptr;
    if (pyobj_.compare_exchange_strong(
            expected, pyobj, std::memory_order_acq_rel)) {
      pyobj_interpreter_.store(interpreter, std::memory_order_release);
      return pyobj;
    } else {
      return expected;
    }
  }

  void init_non_atomic(PyInterpreter* interpreter, PyObject* pyobj) {
    pyobj_interpreter_.store(interpreter, std::memory_order_relaxed);
    pyobj_.store(pyobj, std::memory_order_relaxed);
  }

  // Query the PyObject interpreter.  This may return null if there is no
  // interpreter.
  PyInterpreter* pyobj_interpreter() const {
    return pyobj_interpreter_.load(std::memory_order_acquire);
  }

  PyInterpreter& load_pyobj_interpreter() const {
    auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
    TORCH_INTERNAL_ASSERT(
        interpreter, "cannot access PyObject for Tensor - no interpreter set");
    return *interpreter;
  }

  PyObject* load_pyobj() const {
    return pyobj_.load(std::memory_order_acquire);
  }

  PyObject* _unchecked_untagged_pyobj() const {
    return pyobj_.load(std::memory_order_acquire);
  }

  bool has_unique_reference() const {
    PyObject* pyobj = _unchecked_untagged_pyobj();
    return pyobj != nullptr && load_pyobj_interpreter()->refcnt(pyobj) == 1;
  }
  // Test the interpreter tag.  If tagged for the current interpreter, return
  // a non-nullopt (but possibly null) PyObject.  If (possibly) untagged,
  // returns a nullopt.  If it is definitely invalid, raises an error.
  //
  // If `ignore_hermetic_tls` is false and this function is called from a
  // hermetic context (ie, `HermeticPyObjectTLS::get_state()` is true), then
  // nullopt is returned. If `ignore_hermetic_tls` is true, then the hermetic
  // context is ignored, allowing you to check the interpreter tag of a
  // nonhermetic PyObject from within a hermetic context. This is necessary
  // because there are some cases where the deallocator function of a
  // nonhermetic PyObject is called from within a hermetic context, so it must
  // be properly treated as a nonhermetic PyObject.
  //
  // NB: this lives in header so that we can avoid actually creating the
  // std::optional

  // @todo alban: I'm not too sure what's going on here, we can probably delete
  // it but it's worthwhile making sure
  std::optional<PyObject*> check_pyobj(bool ignore_hermetic_tls = false) const {
    impl::PyInterpreter* interpreter =
        pyobj_interpreter_.load(std::memory_order_acquire);
    if (interpreter == nullptr) {
      return std::nullopt;
    }

    if (!ignore_hermetic_tls && c10::impl::HermeticPyObjectTLS::get_state()) {
      return std::nullopt;
    } else {
      return _unchecked_untagged_pyobj();
    }
  }

 private:
  // This is now always the global interpreter if the PyObject is set.
  // Maybe we can remove this field some day...
  std::atomic<PyInterpreter*> pyobj_interpreter_;

  // The PyObject representing this Tensor or nullptr. Ownership is managed
  // by intrusive_ptr. By the time the PyObjectSlot is destroyed, this
  // reference is already dead.
  std::atomic<PyObject*> pyobj_;
};

} // namespace c10::impl
