# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:23:15 2021

@author: GSung
"""


import unittest

from test import sum

class TestSum(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()