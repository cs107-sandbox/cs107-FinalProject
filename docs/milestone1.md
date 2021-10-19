## Automatic Differentiation Docummentation 
#### Introduction
---------------
The goal of this software is to perform automatic differentiation (AD), a technique crucial to many fields in modern science. AD can help find solutions in complex, nonlinear systems of equations that are impossible to find analytically. For instance, in neural networks, AD allows us to find the optimal combination of weights via gradient-based optimization.

#### Background 
---------------

#### How to use Automatic Differentiation
---------------
We will primarily write a series of functions that are necessary in AD. For instance, calculating the gradient of a function, finding the intersection of two lines, evaluating the derivative at a point of interest x, etc. As an example, finding the gradient will look like:
from AutomaticDifferentiation import gradient as grad
def f(x):
	y = x*x – 4*x + 7
	return y

grad_f = grad(f)
grad_f(2) 
\>> 0.00

#### Software Organization 
---------------
For the structure of the directory, It would be best to keep every class as an independent file for easier access as we continue to integrate and work on the project. Some of the following models will be helpful throughout our project:
main: The environment where top-level code is run. Covers command-line interfaces, import-time behavior
cmath: Mathematical functions for complex numbers.
Decimal: Implementation of the General Decimal Arithmetic Specification
doctest: Test pieces of code within docstrings
fractions: Rational numbers
functools: Higher-order functions and operations on callable objects.
math: Mathematical functions
numbers: Numeric abstract base classes (Complex, Real, Integral, etc.)
test: Regression tests package containing the testing suite for Python

Our test files would all be placed within a test directory. We will attempt to implement TravisCI and Codecov for the AutomaticDifferentiation Software. Since TravisCI is able to upload coverage reports to Codecov, we will be able to check the program's total success. Although we are in the earlier stages of our project, it would be ideal to create a package website that gives users both the information about every function and examples.Since the scale of this project seems to be small, it would be best to structure this software as a python package. 

#### Implementation
---------------
Core data structure: Graph, list
We will implement three parent classes in total (Node, Operation, Execution)
The Node class corresponds to a node in a computation graph.
Class Node:
Attributes:
1.      self.inputs: a list of input nodes that lead to new node
2.      self.operation: specific operation that combines input nodes
3.      self.grad: initial gradient for each node
4.      self.id: unique id for each node
Method:
 
Class Operation:
Method:
1.      __call__: create a new node and associate the operation with that node
2.      compute_value: compute the value of new node given a list of input nodes
3.      compute_gradient: compute gradient contribution to each input node given value of
Children class that inherit Operation: AddOp, AddByConstOp, MulOp, SinOp, CosOp, logOp, ExpOP
Class Executor:
Attributes:
Method:
1.      __init__: initialize a list of nodes whose values need to be computed
2.       run: compute values of node given computation graph in topological order

#### Licensing 
---------------
We plan to adopt the MIT License for its simplicity. AD already has several packages, and our program is unlikely to be monetized.  
