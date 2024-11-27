# test/conftest.py

import sys
import os
import pytest

# Add the 'src' directory to the sys.path for all tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
