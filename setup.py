from setuptools import setup, find_packages

setup(
    name="iabet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "matplotlib",
    ],
    author="IABET Team",
    description="Sistema de predicción de apuestas NBA con IA",
    python_requires=">=3.8",
) 