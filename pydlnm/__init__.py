"""
PyDLNM: Python Distributed Lag Non-linear Models
A Python library for fitting and analyzing DLNMs
"""

__version__ = "0.1.0"
__author__ = "Omar Ahmed"

# Import main components
from .models.dlnm import DLNM
from .core.basis import *
from .core.crossbasis import *
