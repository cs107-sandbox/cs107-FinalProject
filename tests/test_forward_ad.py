

from src.forward_ad import AutoDiffToy
import pytest
import numpy as np

class TestFunctions:
    
    ##########################################
    # __init__
    
    def test_init_value(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiffToy(value, der=derivative, label=label)
        assert x.val == value
    
    def test_init_der(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiffToy(value, der=derivative, label=label)
        assert x.der == derivative
    
    def test_init_label(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiffToy(value, der=derivative, label=label)
        assert x.label == label
    
    ##########################################
    # Addition
    def test_add_val(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x + 3.0
        assert f.val == 5.0        
    
    def test_add_der(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x + 3.0
        assert f.der == 1.0        

    def test_add_label(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x + 3.0
        assert f.label == 'x'  

    def test_radd_val(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 3.0 + x
        assert f.val == 5.0  

    def test_radd_der(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 3.0 + x
        assert f.der == 1.0         

    def test_radd_label(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 3.0 + x
        assert f.label == 'x'  
        
    ##########################################
    # Subtraction
    def test_sub_val(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x - 1.0
        assert f.val == 1.0        
    
    def test_sub_der(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x - 1.0
        assert f.der == 1.0        

    def test_sub_label(self):
        """check that __add__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x - 1.0
        assert f.label == 'x'  

    def test_rsub_val(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 1 - x
        assert f.val == -1.0

    def test_rsub_der(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 1 - x
        assert f.der == -1.0         

    def test_rsub_label(self):
        """check that __radd__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 1 - x
        assert f.label == 'x' 
    
    ##########################################
    # multiplication
    def test_mul_val(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x * 2.0
        assert f.val == 4.0     
    
    def test_mul_der(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x * 2.0
        assert f.der == 2.0

    def test_mul_label(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x * 2.0
        assert f.label == 'x'  

    def test_rmul_val(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 * x
        assert f.val == 4.0 

    def test_rmul_der(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 * x
        assert f.der == 2.0       

    def test_rmul_label(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 * x
        assert f.label == 'x' 

    ##########################################
    # true division
    def test_truediv_val(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x / 2.0
        assert f.val == 1.0
    
    def test_truediv_der(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x / 2.0
        assert f.der == 0.5

    def test_truediv_label(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x / 2.0
        assert f.label == 'x'  

    def test_rtruediv_val(self):
        """check that __rtruediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 / x
        assert f.val == 1.0

    def test_rtruediv_der(self):
        """check that __rtruediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 / x
        assert f.der == -0.5       

    def test_rtruediv_label(self):
        """check that __rtruediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 / x
        assert f.label == 'x' 
       
    ##########################################
    # negation
    def test_neg_val(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = -x
        assert f.val == -2.0
    
    def test_neg_der(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = -x
        assert f.der == -1.0

    def test_neg_label(self):
        """check that __truediv__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = -x
        assert f.label == 'x'  
    
    
    ##########################################
    # power
    
    def test_pow_val(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x ** 2.0
        assert f.val == 4.0     
    
    def test_pow_der(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x ** 2.0
        assert f.der == 4.0

    def test_pow_label(self):
        """check that __mul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x ** 2.0
        assert f.label == 'x'  

    def test_rpow_val(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 ** x
        assert f.val == 4.0 

    def test_rpow_der(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 ** x
        e = 10**(-6)
        
        assert abs(f.der - 2.772588722) < e       

    def test_rpow_label(self):
        """check that __rmul__ has been overwritten correctly"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 ** x
        assert f.label == 'x'
        
    ##########################################
    # sin
    def test_sin_val(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.sin()
        e = 10**(-6)
        
        assert abs(f.val - 1.0) < e
        
    def test_sin_der(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.sin()
        e = 10**(-6)
        
        assert abs(f.der) < e
        
    def test_sin_label(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.sin()
        assert f.label == 'x'


    ##########################################
    # cos
    def test_cos_val(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.cos()
        e = 10**(-6)
        
        assert abs(f.val) < e
        
    def test_cos_der(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.cos()
        e = 10**(-6)
        
        assert abs(f.der + 1.0) < e
        
    def test_cos_label(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 2
        x = AutoDiffToy(value, der=1, label="x")
        f = x.cos()
        assert f.label == 'x'
    
    ##########################################
    # tan
    def test_tan_val(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 4
        x = AutoDiffToy(value, der=1, label="x")
        f = x.tan()
        e = 10**(-6)
        
        assert abs(f.val - 1.0) < e
        
    def test_tan_der(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 4
        x = AutoDiffToy(value, der=1, label="x")
        f = x.tan()
        e = 10**(-6)
        
        assert abs(f.der - 2.0) < e
        
    def test_tan_label(self):
        """check that sin function has been correctly implemented"""
        value = np.pi / 4
        x = AutoDiffToy(value, der=1, label="x")
        f = x.tan()
        assert f.label == 'x'    
    
    
    ##########################################
    # exp
    def test_exp_val(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x.exp()
        e = 10**(-6)
        
        assert abs(f.val - 7.3890561) < e
        
    def test_exp_der(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x.exp()
        e = 10**(-6)
        
        assert abs(f.der - 7.3890561) < e
        
    def test_exp_label(self):
        """check that sin function has been correctly implemented"""
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x.exp()
        assert f.label == 'x'      


#########################
# Check to see exceptions are raised properly
def test_divzero():
    with pytest.raises(Exception) as e_info:
        value = 2.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x / 0 # needs to raise zerodivisionerror

def test_rdivzero():
    with pytest.raises(Exception) as e_info:
        value = 0
        x = AutoDiffToy(value, der=1, label="x")
        f = 2.0 / x # needs to raise zerodivisionerror

def test_powzero():
    with pytest.raises(Exception) as e_info:
        value = 0.0
        x = AutoDiffToy(value, der=1, label="x")
        f = x ** -2

def test_rpowzero():
    with pytest.raises(Exception) as e_info:
        value = -1.0
        x = AutoDiffToy(value, der=1, label="x")
        f = 0 ** x
                
def test_tanzero():
    with pytest.raises(Exception) as e_info:
        value = 0.5*np.pi 
        x = AutoDiffToy(value, der = 1, label = "x")
        f = x.tan()
        

