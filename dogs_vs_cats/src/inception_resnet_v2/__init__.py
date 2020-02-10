"""Top-level package for the 'dltf' framework.

Running ``import dltf`` will recursively import all important subpackages and modules.
"""

import logging

import dogs_vs_cats.src.inception_resnet_v2

logger = logging.getLogger("dogs_vs_cats")

__url__ = "https://github.com/ShiNik/DeepLearning_Tensorflow"
__version__ = "0.1.0"
