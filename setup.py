from setuptools import setup, find_packages

setup(
    name="grid-fed-rl-gym",
    version="1.0.0",
    description="Federated Reinforcement Learning for Power Grid Control",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
