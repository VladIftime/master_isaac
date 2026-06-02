import os
import toml
from setuptools import find_packages, setup

INSTALL_REQUIRES = ["isaaclab"]

EXTENSION_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(EXTENSION_ROOT_DIR, "config", "extension.toml")) as f:
    ext_data = toml.load(f)

setup(
    name=ext_data["package"]["name"],
    version=ext_data["package"]["version"],
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    author=ext_data["package"]["author"],
    maintainer=ext_data["package"]["maintainer"],
    url=ext_data["package"]["website"],
    license="BSD-3-Clause",
    zip_safe=False,
    python_requires=">=3.10",
)
