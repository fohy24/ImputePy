# read version from installed package
from importlib.metadata import version
__version__ = version("imputepy")

from imputepy import LGBMimputer
