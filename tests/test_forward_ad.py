

from src.AutomaticDifferentiation import AutoDiff
from src.AutomaticDifferentiation import ForwardFunctions
import pytest
import numpy as np
from numpy.testing import assert_almost_equal


class TestFunctions:
    
    ##########################################
    # __init__, float
    
    def test_init_value(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.val == value
    
    def test_init_der(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.der == derivative
    
    def test_init_der_list(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = [1.0]
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.der == derivative
        
    def test_init_der_array(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = np.array([1.0])
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.der == derivative

        
    def test_init_label(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.label == [label]
        
        
    def test_init_label_list(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = ['x']
        x = AutoDiff(value, der=derivative, label=label)
        assert x.label == label
        
    ##########################################
    # __init__, array
    
    def test_init_value_array(self):
        """check that variables can be initialized correctly"""
        value = np.array([1,2,3])
        derivative = [1.0, 1.0, 1.0]
        label = ['x','y','z']
        x = AutoDiff(value, der=derivative, label=label)
        assert np.array_equal(x.val, value.reshape(value.shape[0],1))
        
    def test_init_der_2array(self):
        """check that variables can be initialized correctly"""
        value = np.array([2.0, 1.0])
        derivative = np.array([1.0, 1.0])
        label = ['x', 'y']
        x = AutoDiff(value, der=derivative, label=label)
        assert np.array_equal(x.der, np.array(derivative, dtype=np.float64))

    def test_init_der_narray(self):
        """check that variables can be initialized correctly"""
        value = np.array([1,2,3])
        derivative = [1.0, 1.0, 1.0]
        label = ['x','y','z']
        x = AutoDiff(value, der=derivative, label=label)
        assert np.array_equal(x.der, np.array(derivative, dtype=np.float64))
   
    def test_init_label_array(self):
        """check that variables can be initialized correctly"""
        value = np.array([1,2,3])
        derivative = [1.0, 1.0, 1.0]
        label = ['x','y','z']

        x = AutoDiff(value, der=derivative, label=label)
        assert x.label == label
    
    ##########################################
    # function
    def test_init_function_label(self):
        """check that functions can be created correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])
        
        assert f.labels == ['x','y'] 
        
        
    def test_init_function_value(self):
        """check that functions can be created correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])
        
        assert f.values == [[np.sqrt(2)],[np.sqrt(3)]]
    
    def test_init_function_der(self):
        """check that functions can be created correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])

        assert f.jacobians == [[x.sqrt().der[0][0],0],[0, y.sqrt().der[0][0]]]
    
#TODO: write tests for exceptions in this section  
    
    ##########################################
    # Addition
    def test_add_val(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2.0, der=1, label="x")
        y = AutoDiff(3.0, der=1, label="y")
        z = AutoDiff(4.0, der=1, label="z")
        f = ForwardFunctions([x + y, y + z])
        
        assert f.values == [[5.0],[7.0]]        
    
    def test_add_der(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2.0, der=1, label="x")
        y = AutoDiff(3.0, der=1, label="y")
        z = AutoDiff(4.0, der=1, label="z")
        f = ForwardFunctions([x + y, y + z])
        
        assert f.jacobians == [[1.0,1.0,0],[0, 1.0, 1.0]]     

    def test_add_label(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2.0, der=1, label="x")
        y = AutoDiff(3.0, der=1, label="y")
        z = AutoDiff(4.0, der=1, label="z")
        f = ForwardFunctions([x + y, y + z])
        
        assert f.labels == ['x','y','z']     

    def test_radd_val(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 + x, x + y])
        
        assert f.values == [[5.0],[5.0]]  

    def test_radd_der(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 + x, x + y])
        
        assert f.jacobians == [[1.0,0],[1.0,1.0]]  

    def test_radd_label(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 + x, x + y])
        
        assert f.labels == ['x','y']  
        
    ##########################################
    # Subtraction
    def test_sub_val(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x - 3.0, x + y])
        assert f.values == [[-1.0], [5.0]]        
    
    def test_sub_der(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x - 3.0, x + y])
        assert f.jacobians == [[1.0,0], [1.0, 1.0]]        

    def test_sub_label(self):
        """check that __add__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x - 3.0, x + y])
        assert f.labels == ['x','y']        

    def test_rsub_val(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 - x, x + y])
        assert f.values == [[1.0], [5.0]]

    def test_rsub_der(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 - x, x + y])
        assert f.jacobians == [[-1.0,0], [1.0, 1.0]]
   

    def test_rsub_label(self):
        """check that __radd__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3.0 - x, x + y])
        assert f.labels == ['x', 'y']

    
    ##########################################
    # multiplication
    def test_mul_val(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x * 3, x * y, y * x])
        assert f.values == [[6.0],[6.0],[6.0]]
    
    def test_mul_der(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x * 3, x * y, y * x])
        assert f.jacobians == [[3.0,0],[3.0,2.0],[3.0, 2.0]]

    def test_mul_label(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x * 3, x * y, y * x])
        assert f.labels == ['x','y']  

    def test_mul_array(self):
        """check that __mul__ has been overwritten correctly"""
        X = AutoDiff(np.array([2,2]), der=[1,1], label=["x1","x2"])
        Y = AutoDiff(np.array([1,3]), der=[1,1], label=["y1","y2"])
        f = ForwardFunctions([X * 3, Y * 4])
        assert f.values == [[6.0],[4.0]]
#Todo: check if this needs to work with vector*vector or vector+vector
        
    def test_rmul_val(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3 * x, x * y, y * x])
        assert f.values == [[6.0],[6.0],[6.0]]

    def test_rmul_der(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3 * x, x * y, y * x])
        assert f.jacobians == [[3.0,0],[3.0,2.0],[3.0, 2.0]]

    def test_rmul_label(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([3 * x, x * y, y * x])
        assert f.labels == ['x','y']  

    ##########################################
    # true division
    def test_truediv_val(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(4, der=1, label="y")
        f = ForwardFunctions([x / 2, x / y])
        assert f.values == [[1.0],[0.5]]
    
    def test_truediv_der(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(4, der=1, label="y")
        f = ForwardFunctions([x / 2, x / y])
        assert f.jacobians == [[0.5, 0],[0.25, -0.125]]

    def test_truediv_label(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(4, der=1, label="y")
        f = ForwardFunctions([x / 2, x / y])
        assert f.labels == ['x','y'] 

    def test_rtruediv_val(self):
        """check that __rtruediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([2 / x, x + y])
        assert f.values == [[1.0],[5.0]]

    def test_rtruediv_der(self):
        """check that __rtruediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([2 / x, x + y])
        assert f.jacobians == [[-0.5,0],[1.0, 1.0]]

    def test_rtruediv_label(self):
        """check that __rtruediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([2 / x, x + y])
        assert f.labels == ['x','y']
       
    ##########################################
    # negation
    def test_neg_val(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        z = AutoDiff(2, der=1, label="z")
        f = ForwardFunctions([-x / z, x / z])
        
        assert f.values == [[-1.0], [1.0]]
    
    def test_neg_der(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        z = AutoDiff(2, der=1, label="z")
        f = ForwardFunctions([-x / z, x / z])
        
        assert f.jacobians == [[-0.5, 0.5], [0.5, -0.5]]

    def test_neg_label(self):
        """check that __truediv__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        z = AutoDiff(2, der=1, label="z")
        f = ForwardFunctions([-x / z, x / z])
        
        assert f.labels == ['x','z']
    
    ##########################################
    # power
    
    def test_pow_val(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x ** 2, y ** x])
        assert f.values == [[4.0],[9.0]]     
    
    def test_pow_der(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(1, der=1, label="y")
        f = ForwardFunctions([x ** 2, y ** x])
        assert f.jacobians == [[4.0,0],[0.0, 2.0]]

    def test_pow_label(self):
        """check that __mul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(1, der=1, label="y")
        f = ForwardFunctions([x ** 2, y ** x])
        assert f.labels == ['x','y']  

    def test_rpow_val(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([2 ** x, 2 ** (x + y)])
        assert f.values == [[4.0],[32.0]] 

    def test_rpow_der(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(1, der=1, label="x")
        y = AutoDiff(1, der=1, label="y")
        f = ForwardFunctions([2 ** x, 2 ** (x + y)])
        assert f.jacobians == [[np.log(2)*2*1,0],[np.log(2)*4*1.0,np.log(2)*4*1.0]]       

    def test_rpow_label(self):
        """check that __rmul__ has been overwritten correctly"""
        x = AutoDiff(1, der=1, label="x")
        y = AutoDiff(1, der=1, label="y")
        f = ForwardFunctions([2 ** x, 2 ** (x + y)])
        assert f.labels == ['x','y']
        
    ##########################################
    # sin
    def test_sin_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.sin(), (x + y).sin()])
        e = 1e-8
        assert f.values[0] == [1.0] and f.values[1][0] < e
        
    def test_sin_der(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.sin(), (x + y).sin()])
        
        assert_almost_equal(f.jacobians,[[-1,0],[0,0]])        
        
    def test_sin_label(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.sin(), (x + y).sin()])
        assert f.labels == ['x','y']


    ##########################################
    # cos
    def test_cos_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.cos(), (x + y).cos()])

        assert_almost_equal(f.values,[[0],[-1]])        
        
        
    def test_cos_der(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.cos(), (x + y).cos()])

        assert_almost_equal(f.jacobians,[[-1,0],[0,0]])        
        
    def test_cos_label(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.cos(), (x + y).cos()]) 
        assert f.labels == ['x','y']
    
    ##########################################
    # tan
    def test_tan_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi, der=1, label="x")
        y = AutoDiff(np.pi, der=1, label="y")
        f = ForwardFunctions([x.tan(), (x + y).tan()])
        assert_almost_equal(f.values, [[0],[0]])
        
    def test_tan_der(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi, der=1, label="x")
        y = AutoDiff(np.pi, der=1, label="y")
        f = ForwardFunctions([x.tan(), (x + y).tan()])
        
        assert_almost_equal(f.jacobians,[[1,0],[1,1]])
        
        
    def test_tan_label(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(np.pi, der=1, label="x")
        y = AutoDiff(np.pi, der=1, label="y")
        f = ForwardFunctions([x.tan(), (x + y).tan()])
        assert f.labels == ['x','y']    
    
        
    ##########################################
    # sinh
    def test_sinh_val(self):
        """check that sinh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sinh(), y.sinh()])
        assert_almost_equal(f.values, [np.sinh(x.val).tolist()[0],np.sinh(y.val).tolist()[0]])
        
    def test_sinh_der(self):
        """check that sinh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sinh(), y.sinh()])
        xder = np.cosh(x.val).tolist()[0][0]
        yder = np.cosh(y.val).tolist()[0][0]
        
        assert_almost_equal(f.jacobians,[[xder,0],[0,yder]])        
        
    def test_sinh_label(self):
        """check that sinh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sinh(), y.sinh()])
        assert f.labels == ['x','y']


    ##########################################
    # cosh
    def test_cosh_val(self):
        """check that cosh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.cosh(), y.cosh()])
        
        assert_almost_equal(f.values, [np.cosh(x.val).tolist()[0],np.cosh(y.val).tolist()[0]])      
        
        
    def test_cosh_der(self):
        """check that cosh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.cosh(), (x + y).cosh()])
        xder = np.sinh(x.val).tolist()[0][0]
        yder = np.sinh(y.val + x.val).tolist()[0][0]

        assert_almost_equal(f.jacobians,[[xder,0],[yder,yder]])        
        
    def test_cosh_label(self):
        """check that cosh function has been correctly implemented"""
        x = AutoDiff(np.pi / 2, der=1, label="x")
        y = AutoDiff(np.pi / 2, der=1, label="y")
        f = ForwardFunctions([x.cosh(), (x + y).cosh()])
        assert f.labels == ['x','y']
    
    ##########################################
    # tanh
    def test_tanh_val(self):
        """check that tanh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.tanh(), y.tanh()])
        
        assert_almost_equal(f.values, [np.tanh(x.val).tolist()[0],np.tanh(y.val).tolist()[0]])    
        
    def test_tanh_der(self):
        """check that tanh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.tanh(), y.tanh()])
        xder = 1 - np.tanh(x.val).tolist()[0][0]**2
        yder = 1 - np.tanh(y.val).tolist()[0][0]**2
        
        assert_almost_equal(f.jacobians, [[xder, 0],[0,yder]])    
        
        
    def test_tanh_label(self):
        """check that tanh function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.tanh(), y.tanh()])
        assert f.labels == ['x','y']    
        
    ##########################################
    # arcsin
    def test_arcsin_val(self):
        """check that arcsin  function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arcsin(), y.arcsin()])
        
        assert_almost_equal(f.values, [np.arcsin(x.val).tolist()[0],np.arcsin(y.val).tolist()[0]])    
        
    def test_arcsin_der(self):
        """check that arcsin function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arcsin(), y.arcsin()])
        xder = 1 / (1- 0.5**2)**0.5
        yder = 1 / (1- (-0.5)**2)**0.5        
        assert_almost_equal(f.jacobians, [[xder, 0],[0,yder]])    
        
        
    def test_arcsin_label(self):
        """check that arcsin function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arcsin(), y.arcsin()])
        
        assert f.labels == ['x','y']  

   ##########################################
    # arccos
    def test_arccos_val(self):
        """check that arccos function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arccos(), y.arccos()])
        
        assert_almost_equal(f.values, [np.arccos(x.val).tolist()[0],np.arccos(y.val).tolist()[0]])    
        
    def test_arccos_der(self):
        """check that arccos function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arccos(), y.arccos()])
        xder = -1 / (1- 0.5**2)**0.5
        yder = -1 / (1- (-0.5)**2)**0.5        
        assert_almost_equal(f.jacobians, [[xder, 0],[0,yder]])    
        
        
    def test_arccos_label(self):
        """check that arccos function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arccos(), y.arccos()])
        
        assert f.labels == ['x','y']  
          
   ##########################################
    # arctan
    def test_arctan_val(self):
        """check that arcsin  function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arctan(), y.arctan()])        
        assert_almost_equal(f.values, [np.arctan(x.val).tolist()[0],np.arctan(y.val).tolist()[0]])    
        
    def test_arctan_der(self):
        """check that arcsin  function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arctan(), y.arctan()])
        xder = 1 / (1 + 0.5**2)
        yder = 1 / (1 + (-0.5)**2)        
        assert_almost_equal(f.jacobians, [[xder, 0], [0,yder]])    
        
        
    def test_arctan_label(self):
        """check that arcsin function has been correctly implemented"""
        x = AutoDiff(0.5, der=1, label="x")
        y = AutoDiff(-0.5, der=1, label="y")
        f = ForwardFunctions([x.arctan(), y.arctan()])
        
        assert f.labels == ['x','y']  

    ##########################################
    # logistic
#TODO: check function accuracy
    def test_log_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.logistic(), y.logistic()])
        f.values
        
        np.log(2)
        np.log(3)
        
        assert abs(f.val - 7.3890561) < e
        
    def test_log_der(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        f = x.exp()
        e = 10**(-6)
        
        assert abs(f.der - 7.3890561) < e
        
    def test_log_label(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        f = x.exp()
        assert f.label == 'x'    
    
    ##########################################
    # logistic with base
#TODO: check function accuracy
    def test_ln_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.logistic(), y.logistic()])
        f.values
        
        np.log(2)
        np.log(3)
        
        assert abs(f.val - 7.3890561) < e
        
    def test_ln_der(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        f = x.exp()
        e = 10**(-6)
        
        assert abs(f.der - 7.3890561) < e
        
    def test_ln_label(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        f = x.exp()
        assert f.label == 'x'   
        
    ##########################################
    # sqrt
    def test_sqrt_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])
        
        assert_almost_equal(f.values, [[np.sqrt(2)], [np.sqrt(3)]])
        
    def test_sqrt_der(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])
        
        assert_almost_equal(f.jacobians, [[0.5*2**(-0.5),0], [0,0.5*3**(-0.5)]])
        
    def test_sqrt_label(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.sqrt(), y.sqrt()])
        
        assert f.labels == ['x','y']      
    
    ##########################################
    # exp
    def test_exp_val(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.exp(), y.exp()])
        
        assert_almost_equal(f.values, [[np.exp(1)**2], [np.exp(1)**3]])
        
    def test_exp_der(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.exp(), y.exp()])
        
        assert_almost_equal(f.jacobians, [[np.exp(1)**2,0], [0,np.exp(1)**3]])
        
    def test_exp_label(self):
        """check that sin function has been correctly implemented"""
        x = AutoDiff(2, der=1, label="x")
        y = AutoDiff(3, der=1, label="y")
        f = ForwardFunctions([x.exp(), y.exp()])

        assert f.labels == ['x','y']      


#########################
# Check to see exceptions are raised properly
        
def test_init_type():
    with pytest.raises(Exception) as e_info:
        value = 'str'
        x = AutoDiff(value, der = 1, label = 'x') # needs to raise TypeError
        
def test_divzero_ad():
    with pytest.raises(Exception) as e_info:
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        y = AutoDiff(0, der = 1, label = "y")
        f = x / y # needs to raise zerodivisionerror

def test_divzero_int():
    with pytest.raises(Exception) as e_info:
        value = 2.0
        x = AutoDiff(value, der=1, label="x")
        f = x / 0 # needs to raise zerodivisionerror

def test_rdivzero():
    with pytest.raises(Exception) as e_info:
        value = 0
        x = AutoDiff(value, der=1, label="x")
        f = 2.0 / x # needs to raise zerodivisionerror

        
def test_powzero_int():
    with pytest.raises(Exception) as e_info:
        value = 0.0
        x = AutoDiff(value, der=1, label="x")
        f = x ** -2

def test_rpowzero():
    with pytest.raises(Exception) as e_info:
        value = -1.0
        x = AutoDiff(value, der=1, label="x")
        f = 0 ** x
                
def test_tanzero():
    with pytest.raises(Exception) as e_info:
        value = 0.5*np.pi 
        x = AutoDiff(value, der = 1, label = "x")
        f = x.tan()
        
def test_arcsinzero():
    with pytest.raises(Exception) as e_info:
        value = -2
        x = AutoDiff(value, der = 1, label = "x")
        f = x.arcsin()

def test_arccoszero():
    with pytest.raises(Exception) as e_info:
        value = -2
        x = AutoDiff(value, der = 1, label = "x")
        f = x.arccos()

def test_lnzero():
    with pytest.raises(Exception) as e_info:
        value = -1
        x = AutoDiff(value, 1, 'x')
        f = x.ln()

    