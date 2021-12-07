# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:54:46 2021

@author: GSung
"""
import pytest
import numpy as np
from src.ReverseAD import ReverseAD
from src.ReverseAD import ReverseFunctions
from src.ReverseAD import print_var_name


class TestFunctions:
    
    ##########################################
    # __init__
    
    def test_init_value(self):
        """check that variables can be initialized correctly"""
        value = 2.0
        derivative = 1.0
        label = 'x'
        x = AutoDiff(value, der=derivative, label=label)
        assert x.val == value
        
        
        