# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:54:46 2021

@author: GSung
"""
import pytest
import numpy as np
from src.ReverseAD import ReverseAD
from src.ReverseAD import ReverseFunctions

class TestFunctions:
    
    ##########################################
    # __init__
    
    def test_init_value(self):
        """check that variables can be initialized correctly"""
        x = ReverseAD(4)
        assert x.value == 4
        
    def test_init_grad(self):
        """check that variables can be initialized correctly"""
        x = ReverseAD(4)
        assert x.local_gradients == [(None,1)]
        
    ##########################################
    # add
    def test_add_value(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        z = ReverseAD(4, label = "z")        
        f = ReverseFunctions([x + y, y + z + 2], [x,y,z])
        
        assert f.vals == [5,9]
    
    
    def test_add_ders(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        z = ReverseAD(4, label = "z")        
        f = ReverseFunctions([x + y, y + z + 2], [x, y, z])
                
        assert np.array_equal(f.ders, np.array([[1, 1, 0],[0, 1, 1]]))
                
    def test_add_label(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        z = ReverseAD(4, label = "z")        

        f = ReverseFunctions([x + y, y + z + 2], [x, y, z])
        
        assert f.vars == ['x','y','z']

    def test_radd_value(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        z = ReverseAD(4, label = "z")        
        f = ReverseFunctions([3 + x, y + z], [x, y, z])
        
        assert f.vals == [5,7]
    
    
    def test_radd_ders(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        z = ReverseAD(4, label = "z")     
        f = ReverseFunctions([3 + x, y + z], [x, y, z])
                
        assert np.array_equal(f.ders, np.array([[1, 0, 0],[0, 1, 1]]))
        
       
    def test_radd_label(self):
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x + y, 2 + y], [x, y])
        
        assert f.vars == ['x','y']        

    ##########################################
    # Subtraction
    def test_sub_val(self):
        """check that __add__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x - 3.0, x + y])
        assert f.vals == [-1.0, 5]        
    
    def test_sub_der(self):
        """check that __add__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x - 3.0, x + y], [x, y])
        assert np.array_equal(f.ders,np.array([[1, 0],[1,1]]))

    def test_sub_label(self):
        """check that __add__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x - 3.0, x + y], [x, y])
        assert f.vars == ['x','y']        

    def test_rsub_val(self):
        """check that __radd__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3.0 - x, x - y, y - x],[x,y])
        assert np.array_equal(f.vals, [1, -1, 1])

    def test_rsub_der(self):
        """check that __radd__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3.0 - x, x - y, y - x],[x,y])
        assert np.array_equal(f.ders,[[-1,0], [1,-1],[-1, 1]]) 

    def test_rsub_label(self):
        """check that __radd__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3.0 - x, x - y, y - x],[x,y])
        assert f.vars == ['x', 'y']

    ##########################################
    # multiplication
    def test_mul_val(self):
        """check that __mul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x * 3, x * y, y * x], [x,y])
        assert f.vals == [6.0,6.0,6.0]
    
    def test_mul_der(self):
        """check that __mul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x * 3, x * y, y * x], [x,y])
        assert np.array_equal(f.ders, [[3.0,0],[3.0,2.0],[3.0, 2.0]])

    def test_mul_label(self):
        """check that __mul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([x * 3, x * y, y * x], [x,y])
        assert f.vars == ['x','y']  

        
    def test_rmul_val(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3 * x, x * y, y * x], [x,y])
        assert f.vals == [6,6,6]

    def test_rmul_der(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3 * x, x * y, y * x], [x,y])
        assert np.array_equal(f.ders, [[3,0],[3,2],[3, 2]])

    def test_rmul_label(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([3 * x, x * y, y * x], [x,y])
        assert f.vars == ['x','y']  

    ##########################################
    # Inversion
    def test_inv_val(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2,  label="x")
        f = ReverseFunctions([2 / x],[x])
        assert f.vals == [1]
    
    def test_inv_der(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2,  label="x")
        f = ReverseFunctions([2 / x],[x])
        assert np.array_equal(f.ders, [[-0.5]])

    def test_inv_label(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2,  label="x")
        f = ReverseFunctions([2 / x],[x])
        assert f.vars == ['x'] 

    ##########################################
    # true division
    #TODO
    def test_truediv_val(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([x / 2, x / z], [x, z])
        assert f.vals == [1.0,0.5]
    
    def test_truediv_der(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([x / 2, x / z], [x, z])
        assert np.array_equal(f.ders, [[0.5, 0],[0.25, -0.125]])

    def test_truediv_label(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([x / 2, x / z], [x, z])
        assert f.vars == ['x','z'] 
       
    ##########################################
    # negation
    def test_neg_val(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([-x / 2, -z], [x,z])
        assert f.vals == [-1.0, -4]
    
    def test_neg_der(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([-x / 2, -z], [x,z])        
        assert np.array_equal(f.ders, [[-0.5, 0], [0, -1]])

    def test_neg_label(self):
        """check that __truediv__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        z = ReverseAD(4, label = "z")
        f = ReverseFunctions([-x / 2, -z], [x,z])                
        assert f.vars == ['x','z']

    ##########################################
    # power
    
    def test_pow_val(self):
        """check that __pow__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(2, label = "y")
        f = ReverseFunctions([x ** 2, (x + y) ** 2], [x, y])
        assert f.vals == [4, 16]     
    
    def test_pow_der(self):
        """check that __mul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(2, label = "y")
        f = ReverseFunctions([x ** 2, (x + y) ** 2], [x, y])
        assert np.array_equal(f.ders, [[4.0,0],[8, 8]])

    def test_pow_label(self):
        """check that __mul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(2, label = "y")
        f = ReverseFunctions([x ** 2, (x + y) ** 2], [x, y])
        assert f.vars == ['x','y']  

    def test_rpow_val(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(2, label = "y")
        f = ReverseFunctions([2 ** x, 2**(x + y)], [x, y])
        assert f.vals == [4, 16]     
    
    def test_rpow_der(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([2 ** x, 2**(x + y)], [x, y])
        assert np.array_equal(f.ders, [[np.log(2)*2*2,0],[np.log(2)*32,np.log(2)*32]])

    def test_rpow_label(self):
        """check that __rmul__ has been overwritten correctly"""
        x = ReverseAD(2, label = "x")
        y = ReverseAD(3, label = "y")
        f = ReverseFunctions([2 ** x, 2**(x + y)], [x, y])
        assert f.vars== ['x','y']
        
    ##########################################
    # sin
    def test_sin_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.sin(), (x + y).sin()], [x, y])
    
        assert np.allclose(f.vals, [1, 0])
        
    def test_sin_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.sin(), (x + y).sin()], [x, y])
        
        assert np.allclose(f.ders,[[0,0],[-1,-1]])        
        
    def test_sin_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.sin(), (x + y).sin()], [x, y])
        assert f.vars == ['x','y']


    ##########################################
    # cos
    def test_cos_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.cos(), (x + y).cos()], [x, y])
    
        assert np.allclose(f.vals, [0, -1])       
        
        
    def test_cos_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.cos(), (x + y).cos()], [x, y])
        
        assert np.allclose(f.ders,[[-1,0],[0,0]])        
        
    def test_cos_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi / 2, label = "x")
        y = ReverseAD(np.pi / 2, label = "y")
        f = ReverseFunctions([x.cos(), (x + y).cos()], [x, y])
    
        assert f.vars == ['x','y']
    
    ##########################################
    
    # tan
    def test_tan_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi, label = "x")
        y = ReverseAD(np.pi, label = "y")
        f = ReverseFunctions([x.tan(), (x + y).tan()], [x, y])
        
        assert np.allclose(f.vals, [0,0])
        
    def test_tan_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi, label = "x")
        y = ReverseAD(np.pi, label = "y")
        f = ReverseFunctions([x.tan(), (x + y).tan()], [x, y])
        
        assert np.allclose(f.ders,[[1,0],[1,1]])
        
        
    def test_tan_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(np.pi, label = "x")
        y = ReverseAD(np.pi, label = "y")
        f = ReverseFunctions([x.tan(), (x + y).tan()], [x, y])
        assert f.vars == ['x','y']    
    
        
    ##########################################
    # sinh
    def test_sinh_val(self):
        """check that sinh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sinh(), y.sinh()], [x, y])
        assert np.allclose(f.vals, [np.sinh(2),np.sinh(3)])
        
    def test_sinh_der(self):
        """check that sinh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sinh(), y.sinh()], [x, y])
        xder = np.cosh(2)
        yder = np.cosh(3)
        
        assert np.allclose(f.ders,[[xder,0],[0,yder]])        
        
    def test_sinh_label(self):
        """check that sinh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sinh(), y.sinh()], [x, y])
        assert f.vars== ['x','y']
        
    ##########################################
    # cosh
    def test_cosh_val(self):
        """check that cosh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.cosh(), (x+y).cosh()], [x, y])
        assert np.allclose(f.vals, [np.cosh(2),np.cosh(5)])  
        
        
    def test_cosh_der(self):
        """check that cosh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.cosh(), (x+y).cosh()], [x, y])
        xder = np.sinh(2)
        yder = np.sinh(5)

        assert np.allclose(f.ders,[[xder,0],[yder,yder]])        
        
    def test_cosh_label(self):
        """check that cosh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.cosh(), (x+y).cosh()], [x, y])
        assert f.vars == ['x','y']
        
    
    ##########################################
    # tanh
    def test_tanh_val(self):
        """check that tanh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.tanh(), (x+y).tanh()], [x, y])
        
        assert np.allclose(f.vals, [np.tanh(2), np.tanh(5)])    
        
    def test_tanh_der(self):
        """check that tanh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.tanh(), y.tanh()], [x, y])
        xder = 1 - np.tanh(2)**2
        yder = 1 - np.tanh(3)**2
        
        assert np.allclose(f.ders, [[xder, 0],[0,yder]])    
        
        
    def test_tanh_label(self):
        """check that tanh function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.tanh(), y.tanh()], [x, y])
        assert f.vars == ['x','y']    
        
    ##########################################
    # arcsin
    def test_arcsin_val(self):
        """check that arcsin  function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arcsin(), y.arcsin()], [x, y])
        
        assert np.allclose(f.vals, [np.arcsin(0.5),np.arcsin(-0.5)])    
        
    def test_arcsin_der(self):
        """check that arcsin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arcsin(), y.arcsin()], [x, y])
        xder = 1 / (1- 0.5**2)**0.5
        yder = 1 / (1- (-0.5)**2)**0.5        
        assert np.allclose(f.ders, [[xder, 0],[0,yder]])    
        
        
    def test_arcsin_label(self):
        """check that arcsin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arcsin(), y.arcsin()], [x, y])
        
        assert f.vars == ['x','y']  

   ##########################################
    # arccos
    def test_arccos_val(self):
        """check that arccos function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arccos(), y.arccos()], [x, y])
        
        assert np.allclose(f.vals, [np.arccos(x.value),np.arccos(y.value)])    
        
    def test_arccos_der(self):
        """check that arccos function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arccos(), y.arccos()], [x, y])
        xder = -1 / (1- 0.5**2)**0.5
        yder = -1 / (1- (-0.5)**2)**0.5        
        assert np.allclose(f.ders, [[xder, 0],[0,yder]])    
        
        
    def test_arccos_label(self):
        """check that arccos function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arccos(), y.arccos()], [x, y])
        
        assert f.vars == ['x','y']  
          
   ##########################################
    # arctan
    def test_arctan_val(self):
        """check that arcsin  function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arctan(), y.arctan()], [x, y])       
        assert np.allclose(f.vals, [np.arctan(x.value),np.arctan(y.value)])    
        
    def test_arctan_der(self):
        """check that arcsin  function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arctan(), y.arctan()], [x, y])
        xder = 1 / (1 + 0.5**2)
        yder = 1 / (1 + (-0.5)**2)        
        assert np.allclose(f.ders, [[xder, 0], [0,yder]])    
        
        
    def test_arctan_label(self):
        """check that arcsin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(-0.5, label="y")
        f = ReverseFunctions([x.arctan(), y.arctan()], [x, y])
        
        assert f.vars == ['x','y']  

    ##########################################
    # logistic
    def test_logistic_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(0.8, label="y")
        f = ReverseFunctions([x.logistic(), y.logistic()], [x, y])
        
        xval = 1/ (1 + np.exp(-0.5))
        yval = 1/ (1 + np.exp(-0.8))
        
        assert np.allclose(f.vals, [xval, yval])
        
    def test_logistic_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(0.8, label="y")
        f = ReverseFunctions([x.logistic(), y.logistic()], [x, y])
        
        xval = 1/ (1 + np.exp(-0.5))
        yval = 1/ (1 + np.exp(-0.8))
        
        xder = xval * (1-xval)
        yder = yval * (1-yval)
        
        assert np.allclose(f.ders, [[xder, 0],[0, yder]])
        
    def test_logistic_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(0.5, label="x")
        y = ReverseAD(0.8, label="y")
        f = ReverseFunctions([x.logistic(), y.logistic()], [x, y])
        assert f.vars ==  ['x','y']     
    
    ##########################################
    # log 
    def test_ln_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.ln(), y.ln()], [x, y])
        assert np.allclose(f.vals,[np.log(2),np.log(3)])
        
    def test_ln_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.ln(), y.ln()], [x, y])
                
        assert np.allclose(f.ders,[[0.5,0],[0, 1/3]])
        
    def test_ln_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.ln(), y.ln()], [x, y])
        assert f.vars ==  ['x','y']     
        
    ##########################################
    # log with base
    def test_lnb_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(8, label="x")
        y = ReverseAD(np.exp(1), label="y")
        f = ReverseFunctions([x.ln_base(2), y.ln_base(np.exp(1))], [x, y])
        
        assert np.allclose(f.vals,[3.0,1.0])
        
    def test_lnb_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(8, label="x")
        y = ReverseAD(np.exp(1), label="y")
        f = ReverseFunctions([x.ln_base(2), y.ln_base(np.exp(1))], [x, y])
        
        assert np.allclose(f.ders, [[1 / (8*np.log(2)), 0], [0,1 /np.exp(1)]])
        
    def test_lnb_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(8, label="x")
        y = ReverseAD(np.exp(1), label="y")
        f = ReverseFunctions([x.ln_base(2), y.ln_base(np.exp(1))], [x, y])
        
        assert f.vars ==  ['x','y']     
  
        
    ##########################################
    
    # sqrt
    def test_sqrt_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sqrt(), y.sqrt()], [x, y])
        
        assert np.allclose(f.vals, [np.sqrt(2), np.sqrt(3)])
        
    def test_sqrt_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sqrt(), y.sqrt()], [x, y])
        
        assert np.allclose(f.ders, [[0.5*2**(-0.5),0], [0,0.5*3**(-0.5)]])
        
    def test_sqrt_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.sqrt(), y.sqrt()], [x, y])
        
        assert f.vars == ['x','y']      
    
    ##########################################
    # exp
    def test_exp_val(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.exp(), y.exp()], [x, y])
        
        assert np.allclose(f.vals, [np.exp(1)**2, np.exp(1)**3])
        
    def test_exp_der(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.exp(), y.exp()], [x, y])
        
        assert np.allclose(f.ders, [[np.exp(1)**2,0], [0,np.exp(1)**3]])
        
    def test_exp_label(self):
        """check that sin function has been correctly implemented"""
        x = ReverseAD(2, label="x")
        y = ReverseAD(3, label="y")
        f = ReverseFunctions([x.exp(), y.exp()], [x, y])

        assert f.vars == ['x','y']      


    #########################

      
def test_init_type():
    with pytest.raises(Exception) as e_info:
        value = 'str'
        x = ReverseAD(value, label = 'x') # needs to raise TypeError
        
def test_divzero_ad():
    with pytest.raises(Exception) as e_info:
        value = 2.0
        x = ReverseAD(value, label="x")
        y = ReverseAD(0, label = "y")
        f = ReverseFunctions(x / y, [x,y]) # needs to raise zerodivisionerror

def test_divzero_int():
    with pytest.raises(Exception) as e_info:
        value = 2.0
        x = ReverseAD(value, label="x")
        f = ReverseFunctions(x / 0, [x]) # needs to raise zerodivisionerror

def test_rdivzero():
    with pytest.raises(Exception) as e_info:
        value = 0
        x = ReverseAD(value, label="x")
        f = ReverseFunctions(2.0 / x, [x]) # needs to raise zerodivisionerror

        
def test_powzero_int():
    with pytest.raises(Exception) as e_info:
        value = 0.0
        x = ReverseAD(value, label="x")
        f = ReverseFunctions(x ** -2, [x])

def test_rpowzero():
    with pytest.raises(Exception) as e_info:
        value = -1.0
        x = ReverseAD(value, label="x")
        f = ReverseFunctions(0 ** x, [x])
                
def test_tanzero():
    with pytest.raises(Exception) as e_info:
        value = 0.5*np.pi 
        x = ReverseAD(value, label = "x")
        y = ReverseAD(np.pi, label = "y")

        f = ReverseFunctions([x.tan(), y.tan()], [x,y])
        
def test_arcsinzero():
    with pytest.raises(Exception) as e_info:
        value = -2
        x = ReverseAD(value, label = "x")
        y = ReverseAD(np.pi, label = "y")
        f = ReverseFunctions([x.arcsin(), y.arcsin()], [x,y])

def test_arccoszero():
    with pytest.raises(Exception) as e_info:
        value = -2
        x = ReverseAD(value, label = "x")
        y = ReverseAD(np.pi, label = "y")
        f = ReverseFunctions([x.arccos(), y.arccos()], [x,y])

def test_lnmin():
    with pytest.raises(Exception) as e_info:
        value = -1
        x = ReverseAD(value, label = 'x')
        y = ReverseAD(5, label = 'y')
        f = ReverseFunctions([x.ln(), y.ln()], [x,y])

def test_lnzero():
    with pytest.raises(Exception) as e_info:
        value = 0
        x = ReverseAD(value, label = 'x')
        y = ReverseAD(5, label = 'y')
        f = ReverseFunctions([x.ln(), y.ln()], [x,y])       
        
         
def test_lnbasezero():    
    with pytest.raises(Exception) as e_info:    
        value = 2
        x = ReverseAD(value, label = 'x')
        y = ReverseAD(5, label = 'y')
        f = ReverseFunctions([x.ln_base(0), y.ln_base(0)], [x,y]) 