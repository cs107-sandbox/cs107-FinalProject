import numpy as np

class AutoDiffToy():
    def __init__(self, val, der=1.0, label=""):
        self.val = val
        self.der = der
        self.label = label

    def __add__(self,other):
        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
            return AutoDiffToy(new_val, new_der, self.label)

        except AttributeError:
            return AutoDiffToy(self.val + other, self.der, self.label)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__radd__(-other)

    def __mul__(self,other):
        try:
            new_val = self.val*other.val
            new_der = self.val*other.der + self.der*other.val
            return AutoDiffToy(new_val, new_der, self.label)

        except AttributeError:
            return AutoDiffToy(self.val*other, self.der*other, self.label)

    def __rmul__(self,other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError

        new_val = self.val/other
        new_der = self.der/other

        return AutoDiffToy(new_val, new_der, self.label)

    def __rtruediv__(self, other):
        if self.val == 0:
            raise ZeroDivisionError

        new_val - other/self.new_val
        new_der = (-other/(self.val**2))*self.der
        return AutoDiffToy(new_val, new_der, self.label)

    def __neg__(self):
        return AutoDiffToy(-self.val, -self.der, self.label)

    def __pow__(self,n):
        if n < 0 and self.val == 0:
            raise ZeroDivisionError

        new_val = self.val**n
        new_der = n*self.val**(n-1)*self.der
        return AutoDiffToy(new_val, new_der, self.label)

    def __rpow__(self,other):
        if other == 0 and self.val < 0:
            raise ZeroDivisionError

        new_val = other ** self.val
        new_der = np.log(other)*new_val*self.der

        return AutoDiffToy(new_val, new_der, self.label)

    def sin(self):
        new_val = np.sin(self.val)
        new_der = np.cos(self.val)*self.der

        return AutoDiffToy(new_val, new_der, self.label)

    def cos(self):
        new_val = np.cos(self.val)
        new_der = -np.sin(self.val)* self.der

        return AutoDiffToy(new_val, new_der, self.label)

    def tan(self):
        if (self.val/np.pi-0.5)%1 == 0.00:
            raise ValueError('Value can not be applied to tangent')

        new_val = np.tan(self.val)
        new_der = np.multiply(1/np.power(np.cos(self.val),2), self.der)

        return AutoDiffToy(new_val, new_der, self.label)

    def exp(self):
        return self.__rpow__(np.exp(1))


# Test
a = 2.0
x = AutoDiffToy(a, der=1, label="x")

alpha = 2.0
beta = 3.0
f = alpha * x + beta
print(f.val, f.der)

f = x * alpha + beta
print(f.val, f.der)

f = beta + alpha * x
print(f.val, f.der)

f = beta + x * alpha
print(f.val, f.der)

f = x ** beta
print(f.val, f.der)

f = beta ** x
print(f.val, f.der)

f = x.sin()
print(f.val, f.der)

f = x.cos()
print(f.val, f.der)

f = x.tan()
print(f.val, f.der)

f = x.exp()
print(f.val, f.der)
