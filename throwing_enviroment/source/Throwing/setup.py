# (c) Open source contributors
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup

setup(
    name="Throwing",
    version="1.0.0",
    packages=["Throwing", "Throwing.tasks", "Throwing.tasks.throwing", "Throwing.tasks.throwing.agents"],
    package_dir={"": "."},
)
