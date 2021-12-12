# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from setuptools import find_packages

import pathlib

_CURRENT_DIR = pathlib.Path(__file__).parent

setup(name="brepnet",
      version="0.1.0",
      url="https://github.com/d61h6k4/brepnet",
      license="Apache 2.0",
      description="BRepNet implementation",
      long_description=(_CURRENT_DIR / "README.md").read_text(),
      long_description_content_type="text/markdown",
      keywords="CAD BRep Neural Network",
      packages=find_packages(),
      install_requires=[
          "absl-py", "numpy", "tensorflow", "tensorflow-datasets", "jax",
          "flax", "jraph", "clu", "ml-collections", "optax", "chex",
          "tfds-fusion-360-gallery-dataset"
      ],
      python_requires=">=3.8",
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ])
