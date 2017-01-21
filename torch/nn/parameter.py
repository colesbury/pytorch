from torch.autograd import Variable


class Parameter(Variable):

    def __new__(cls, data, requires_grad=True):
        return super(Parameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __deepcopy__(self, memo):
        result = type(self)(self.data.clone(), self.requires_grad)
        memo[id(self)] = result
        return result

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()

    def __reduce_ex__(self, proto):
        newargs = (self.data, self.requires_grad)
        state = (self._grad, self._backward_hooks)
        return type(self), newargs, state
