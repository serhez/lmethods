"""
A collection of both novel and baseline methods to enhance the capabilities of language models.
"""

from .method import Method
from .reasoning import MetaPrompting

__all__ = ["Method", "MetaPrompting"]
