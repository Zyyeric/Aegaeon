from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="quick_model_loader",
    version="0.0.2",
    packages=find_packages(include=["quick_model_loader*"]),
    install_requires=[
        "torch>=2.0.0",
        "safetensors>=0.4.3",
        "numpy",
        "pytest",
        "cupy-cuda12x",
    ],
    rust_extensions=[
        RustExtension(
            "quick_model_loader._rlib",
            # ^-- The last part of the name (e.g. "_lib") has to match lib.name
            #     in Cargo.toml and the function name in the `.rs` file,
            #     but you can add a prefix to nest it inside of a Python package.
            path="Cargo.toml",  # Default value, can be omitted
            binding=Binding.PyO3,  # Default value, can be omitted
        )
    ],
    python_requires=">=3.8",
)
