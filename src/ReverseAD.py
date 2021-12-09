import numpy as np
from collections import defaultdict
from numpy.lib.arraysetops import isin

# convert variable name to string


# def print_var_name(variable):
#     for name in globals():
#         if eval(name) == variable:
#             return name

#     return None


class ReverseAD:

    node_dict = {}

    def __init__(self, value, local_gradients=[], label=None):
        """

        -- Parameters
        value : the value of the variable
        local_gradients: the variable's children and corresponding local derivatives

        """
        self.value = value
        if len(local_gradients) == 0:
            self.local_gradients = [(None, 1)]
        else:
            self.local_gradients = local_gradients

        if label is not None:
            ReverseAD.node_dict[self] = label

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return ReverseAD(self.value + other, [(self, 1)])
        elif isinstance(other, ReverseAD):
            value = self.value + other.value
            local_gradients = (
                (self, 1),
                (other, 1)
            )
            return ReverseAD(value, local_gradients)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return ReverseAD(self.value * other, [(self, other)])
        elif isinstance(other, ReverseAD):
            value = self.value * other.value
            local_gradients = (
                (self, other.value),
                (other, self.value)
            )
            return ReverseAD(value, local_gradients)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        value = -1 * self.value
        local_gradients = (
            (self, -1),
        )
        return ReverseAD(value, local_gradients)

    def inv(self):
        """
        Perform inversion(Helper Function)
        -- Parameters
        -- Return
        An ReverseAD object with calculated values, variable’s children and local derivatives.
        -- Demo
        >>> x = ReverseAD(2)
        >>> f = ReverseFunctions([1 / x], [x])
        >>> f.vals
        [0.5]
        >>> f.ders
        [[-0.25]]
        >>> f.vars
        [‘x’]
        """
        value = 1. / self.value
        local_gradients = (
            (self, -1 / self.value**2),
        )
        return ReverseAD(value, local_gradients)
    
    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):

            return self.__mul__(1/other)
        elif isinstance(other, ReverseAD):
            return self.__mul__(other.inv())

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            value = other / self.value
            local_gradients = (
                (self, -other/(self.value ** 2)),
            )
            return ReverseAD(value, local_gradients)
        else:
            return self.__truediv__(other)

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return ReverseAD(self.value ** other, [(self, other * self.value ** (other - 1))])
        elif isinstance(other, ReverseAD):
            value = self.value ** other.value
            local_gradients = (
                (self, other.value * self.value ** (other.value - 1)),
                (other, value * np.log(self.value))
            )
            return ReverseAD(value, local_gradients)

    def __rpow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            value = other ** self.value
            local_gradients = (
                (self, value * np.log(other)),
            )
            return ReverseAD(value, local_gradients)
        else:
            return self.__pow__(other)

    def sin(self):
        value = np.sin(self.value)
        local_gradients = (
            (self, np.cos(self.value)),
        )
        return ReverseAD(value, local_gradients)

    def cos(self):
        value = np.cos(self.value)
        local_gradients = (
            (self, -np.sin(self.value)),
        )
        return ReverseAD(value, local_gradients)

    def tan(self):
        value = np.tan(self.value)
        local_gradients = (
            (self, 1 / np.power(np.cos(self.value), 2)),
        )
        return ReverseAD(value, local_gradients)

    def exp(self):
        """
        Perform the exponential

        -- Parameters

        -- Return
        An ReverseAD object with calculated values, variable's children and local derivatives.

        -- Demo

        >>> x = ReverseAD(2)
        >>> y = ReverseAD(3)
        >>> f = ReverseFunctions([x.exp(), y.exp()], [x, y])
        >>> f.vals
        [7.389, 20.086] 
        >>> f.ders
        [[7.389, 0]
         [0, 20.1]] 
        >>> f.vars
        ['x', 'y']
        """

        value = np.exp(self.value)
        local_gradients = (
            (self, np.exp(self.value)),
        )
        return ReverseAD(value, local_gradients)

    def ln(self):
        value = np.log(self.value)
        local_gradients = (
            (self, 1. / self.value),
        )
        return ReverseAD(value, local_gradients)

    def ln_base(self, base):
        return self.ln() / np.log(base)

    def sinh(self):
        value = np.sinh(self.value)
        local_gradients = (
            (self, np.cosh(self.value)),
        )
        return ReverseAD(value, local_gradients)

    def cosh(self):
        value = np.cosh(self.value)
        local_gradients = (
            (self, np.sinh(self.value)),
        )
        return ReverseAD(value, local_gradients)

    def tanh(self):
        value = np.tanh(self.value)
        local_gradients = (
            (self, 1 - value ** 2),
        )
        return ReverseAD(value, local_gradients)


    def arcsin(self):
        """
        Perform the arcsine

        -- Parameters

        -- Return
        An ReverseAD object with calculated values, variable's children and local derivatives.

        -- Demo

        >>> x = ReverseAD(0.5)
        >>> y = ReverseAD(-0.5)
        >>> f = ReverseFunctions([x.arcsin(), y.arcsin()], [x, y])
        >>> f.vals
        [0.524], -0.524]
        >>> f.ders
        [[1.155, 0]
         [0, 1.155]]
        >>> f.vars
        ['x', 'y']
        """
        if self.value <= -1 or self.value >= 1:
            raise ValueError("Arcsine cannot be applied to this value")
        value = np.arcsin(self.value)
        local_gradients = (
            (self, 1 / (1 - self.value ** 2) ** 0.5),
        )
        return ReverseAD(value, local_gradients)

    def arccos(self):
        """
        Perform the arccosine

        -- Parameters

        -- Return
        An ReverseAD object with calculated values, variable's children and local derivatives.

        -- Demo

        >>> x = ReverseAD(0.5)
        >>> y = ReverseAD(-0.5)
        >>> f = ReverseFunctions([x.arccos(), y.arccos()], [x, y])
        >>> f.vals
        [1.047,2.094]
        >>> f.ders
        [[-1.155, 0]
         [0, -1.155]]
        >>> f.vars
        ['x', 'y']
        """
        if self.value <= -1 or self.value >= 1:
            raise ValueError("Arccosine cannot be applied to this value")
        value = np.arccos(self.value)
        local_gradients = (
            (self, -1 / (1 - self.value ** 2) ** 0.5),
        )
        return ReverseAD(value, local_gradients)
   

    def arctan(self):
        """
        Perform the arctangent

        -- Parameters

        -- Return
        An ReverseAD object with calculated values, variable's children and local derivatives.

        -- Demo

        >>> x = ReverseAD(0.5)
        >>> y = ReverseAD(-0.5)
        >>> f = ReverseFunctions([x.arctan(), y.arctan()], [x, y])
        >>> f.vals
        [0.464, -0.464]
        >>> f.ders
        [[0.8, 0]
         [0, 0.8]]
        >>> f.vars
        ['x', 'y']
        """
        value = np.arctan(self.value)
        local_gradients = (
            (self, 1 / (1 + self.value ** 2)),
        )
        return ReverseAD(value, local_gradients)

    def logistic(self):
        value = 1 / (1 + np.exp(-self.value))
        local_gradients = (
            (self, value * (1 - value)),
        )
        return ReverseAD(value, local_gradients)

    def sqrt(self):
        return self.__pow__(1/2)

    def get_gradients(self):
        """ Compute the first derivatives of `variable`
        with respect to child variables.
        """
        gradients = defaultdict(lambda: 0)

        def compute_gradients(self, path_value):

            for child_variable, local_gradient in self.local_gradients:
                # "Multiply the edges of a path":
                value_of_path_to_child = path_value * local_gradient
                # "Add together the different paths":
                if child_variable == None:
                    # Escape condition (reach leaf nodes)
                    gradients[self] = 1 * value_of_path_to_child
                else:

                    gradients[child_variable] += value_of_path_to_child
                # recurse through graph:
                    compute_gradients(child_variable, value_of_path_to_child)

        compute_gradients(self, path_value=1)
        # (path_value=1 is from `variable` differentiated w.r.t. itself)

        return gradients


class ReverseFunctions():
    def __init__(self, functions, variables=[]):

        values = []
        for function in functions:
            try:
                values.append(function.value)
            except AttributeError:
                values.append(function)  # constant

        all_der = []
        for function in functions:
            curr_der = []
            curr_grad = function.get_gradients()
            for var in variables:
                if var not in curr_grad:
                    curr_der.append(0)
                    continue
                curr_der.append(curr_grad[var])

            all_der.append(curr_der)

        # variable_names = [print_var_name(var) for var in variables]
        variable_names = [ReverseAD.node_dict[var] for var in variables]
        self.vals = values
        self.vars = variable_names
        self.ders = np.array(all_der)


x = ReverseAD(2, label="x")
y = ReverseAD(3, label="y")
z = ReverseAD(4, label="z")

# f = ReverseFunctions([x/2, x/z, z/x], [x, y, z])

f = ReverseFunctions([x - 3.0, x + y], [x, y])
print(f.vals)
print(f.ders)
print(f.vars)

# print(np.tan(2))
