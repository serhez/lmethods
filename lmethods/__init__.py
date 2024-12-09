"""
A collection of both novel and baseline methods to enhance the capabilities of language models.
"""

from .method import Method
from .reasoning import MetaPrompting, RecursivePrompting, ToT

__all__ = ["Method", "MetaPrompting", "RecursivePrompting", "ToT"]
