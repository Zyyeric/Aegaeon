import io
import os
from typing import List

from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


setup(
    name="aegaeon",
    version="0.1.0",
    author="Alibaba Team",
    description="Aegaeon: A multi-LLM inference engine that multiplexes requests at the token level.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=["aegaeon"],
    python_requires=">=3.10",
    install_requires=get_requirements(),
    ext_modules=[
        CUDAExtension(name="aegaeon.ops", sources=["ops/ops.cpp", "ops/swap.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
