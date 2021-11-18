import numpy as np


class AutoDiffToy():
    def __init__(self, val, der=1.0, label=""):
        """
        Constructs an AD object

        -- Parameters
        value : value of input variables 
        der :  derivatives with respect to each variable, default is set to 1.0.
        label : variable name, default is set to "".

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 2.0 * x + 3.0
        >>> f.val 
        7.0
        >>> f.der
        2.0
        >>> f.label
        'x'
        """
        self.val = val
        self.der = der
        self.label = label

    def __add__(self, other):
        """
        Perform addition 

        -- Parameters
        other : values to be added

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x + 3.0
        >>> f.val 
        5.0
        >>> f.der
        1.0
        >>> f.label
        'x'
        """

        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
            return AutoDiffToy(new_val, new_der, self.label)
        except AttributeError:
            return AutoDiffToy(self.val + other, self.der, self.label)

    def __radd__(self, other):
        """
        Perform reverse addition 

        -- Parameters
        other : values to be added

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 3.0 + x
        >>> f.val 
        5.0
        >>> f.der
        1.0
        >>> f.label
        'x'
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Perform subtraction

        -- Parameters
        other : values to be subtracted from self

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x - 1
        >>> f.val 
        1.0
        >>> f.der
        1.0
        >>> f.label
        'x'
        """
        try:
            new_val = self.val - other.val
            new_der = self.der - other.der
            return AutoDiffToy(new_val, new_der, self.label)
        except AttributeError:
            return AutoDiffToy(self.val - other, self.der, self.label)

    def __rsub__(self, other):
        """
        Perform reverse subtraction

        -- Parameters
        other : values from which self is substracted

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 1 - x
        >>> f.val 
        -1.0
        >>> f.der
        -1.0
        >>> f.label
        'x'
        """
        if isinstance(other, float) or isinstance(other, int):
            return AutoDiffToy(other - self.val, -self.der, self.label)
        elif isinstance(other, AutoDiffToy):
            return AutoDiffToy(other.val - self.val, -self.der, self.label)

    def __mul__(self, other):
        """
        Perform multiplication

        -- Parameters
        other : values to be multiplied to self

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x * 2.0
        >>> f.val 
        4.0
        >>> f.der
        2.0
        >>> f.label
        'x'
        """
        try:
            new_val = self.val * other.val
            new_der = self.val * other.der + self.der * other.val
            return AutoDiffToy(new_val, new_der, self.label)
        except AttributeError:
            return AutoDiffToy(self.val * other, self.der * other, self.label)

    def __rmul__(self, other):
        """
        Perform reverse multiplication

        -- Parameters
        other : values to be multiplied to self

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 2.0 * x
        >>> f.val 
        4.0
        >>> f.der
        2.0
        >>> f.label
        'x'
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Perform true division

        -- Parameters
        other : values to divide self

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x / 2.0
        >>> f.val 
        1.0
        >>> f.der
        0.5
        >>> f.label
        'x'
        """
        if other == 0:
            raise ZeroDivisionError
        new_val = self.val / other
        new_der = self.der / other
        return AutoDiffToy(new_val, new_der, self.label)

    def __rtruediv__(self, other):
        """
        Perform reverse true division

        -- Parameters
        other : values to be divided by self

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 2.0 / x
        >>> f.val 
        1.0
        >>> f.der
        -0.5
        >>> f.label
        'x'
        """
        if self.val == 0:
            raise ZeroDivisionError

        new_val = other / self.val
        new_der = (-other/(self.val ** 2)) * self.der
        return AutoDiffToy(new_val, new_der, self.label)

    def __neg__(self):
        """
        Perform negation

        -- Parameters

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = -x
        >>> f.val 
        -2.0
        >>> f.der
        -1.0
        >>> f.label
        'x'
        """
        return AutoDiffToy(-self.val, -self.der, self.label)

    def __pow__(self, n):
        """
        Perform the power of n

        -- Parameters
        n : exponent to which self is raised

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x ** 2
        >>> f.val 
        4.0
        >>> f.der
        4.0
        >>> f.label
        'x'
        """
        if n < 0 and self.val == 0:
            raise ZeroDivisionError

        if n < 1 and self.val < 0:
            raise
        new_val = self.val ** n
        new_der = n * self.val ** (n-1) * self.der
        return AutoDiffToy(new_val, new_der, self.label)

    def __rpow__(self, other):
        """
        Raise a number to the power of self

        -- Parameters
        other : number to be raised

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2.0  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = 2 ** x
        >>> f.val 
        4.0
        >>> f.der
        2.772588722
        >>> f.label
        'x'
        """
        if other == 0 and self.val < 0:
            raise ZeroDivisionError
        new_val = other ** self.val
        new_der = np.log(other) * new_val * self.der

        return AutoDiffToy(new_val, new_der, self.label)

    def sin(self):
        """
        Perform the sine

        -- Parameters

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = np.pi / 2  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x.sin()
        >>> f.val 
        1.0
        >>> f.der
        0.0
        >>> f.label
        'x'
        """
        new_val = np.sin(self.val)
        new_der = np.cos(self.val) * self.der

        return AutoDiffToy(new_val, new_der, self.label)

    def cos(self):
        """
        Perform the cosine

        -- Parameters

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = np.pi / 2  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x.cos()
        >>> f.val 
        0.0
        >>> f.der
        -1.0
        >>> f.label
        'x'
        """
        new_val = np.cos(self.val)
        new_der = -np.sin(self.val) * self.der
        return AutoDiffToy(new_val, new_der, self.label)

    def tan(self):
        """
        Perform the tagent

        -- Parameters

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = np.pi / 4  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x.tan()
        >>> f.val 
        1.0
        >>> f.der
        2.0
        >>> f.label
        'x'
        """
        if (self.val / np.pi - 0.5) % 1 == 0.00:
            raise ValueError("Tangent cannot be applied to this value")
        new_val = np.tan(self.val)
        new_der = np.multiply(1 / np.power(np.cos(self.val), 2), self.der)
        return AutoDiffToy(new_val, new_der, self.label)

    def exp(self):
        """
        Perform the exponential

        -- Parameters

        -- Return
        An AD object with calculated values, derivatives and variable names.

        -- Demo
        # Scalar input (x)

        >>> a = 2  # Value to evaluate at
        >>> x = AutoDiffToy(a, der=1, label="x")
        >>> f = x.exp()
        >>> f.val 
        7.3890561
        >>> f.der
        7.3890561
        >>> f.label
        'x'
        """

        return self.__rpow__(np.exp(1))


# value = 2.0
# x = AutoDiffToy(value, der=1, label="x")
# f = 1 - x
# print(f.val, f.der)
# assert f.val == -1.0


# a = 2.0  # Value to evaluate at
# x = AutoDiffToy(a, der=1, label="x")

# alpha = 2.0
# beta = 3.0
# f = alpha * x + beta
# print(f.val, f.der)

# f = x * alpha + beta
# print(f.val, f.der)

# f = beta + alpha * x
# print(f.val, f.der)

# f = beta + x * alpha
# print(f.val, f.der)

# f = x ** beta
# print(f.val, f.der)

# f = beta ** x
# print(f.val, f.der)

# f = x.sin()
# print(f.val, f.der)

# f = x.cos()
# print(f.val, f.der)

# f = x.tan()
# print(f.val, f.der)

# f = x.exp()
# print(f.val, f.der)
