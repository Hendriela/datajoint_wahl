#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 05/10/2021 11:24
@author: hheise

A few small helper functions.
"""

from typing import List
import re

def alphanumerical_sort(x: List[str]) -> List[str]:
    """
    Sorts a list of strings alpha-numerically, with proper interpretation of numbers in the string. Usually used to sort
    file names to ensure files are processed in the correct order (if file name numbers are not zero-padded).
    Elements are first  sorted by characters. If characters are the same, elements are sorted by numbers in the string.
    Consecutive digits are treated as one number and sorted accordingly (e.g. "text_11" will be sorted behind "text_2").
    Leading Zeros are ignored.

    Args:
        x:  List to be sorted. Elements are usually file names.

    Returns:
        Sorted list.
    """
    x_sort = x.copy()
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]
    x_sort.sort(key=natural_keys)
    return x_sort