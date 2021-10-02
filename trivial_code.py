# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:48:00 2021

@author: GSung
"""


def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"
    
    
if __name__ == "__main__":
    test_sum()
    print("Everything passed")