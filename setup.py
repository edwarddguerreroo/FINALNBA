from setuptools import setup, find_packages

setup(
    name="nba_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "plotly>=5.1.0",
        "shap>=0.39.0",
        "torch>=1.9.0",
        "python-dateutil>=2.8.2",
        "tqdm>=4.61.2",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.3",
        "kaleido>=0.2.1",
        "nbformat>=5.1.3"
    ],
    author="Sistema de Predicción NBA",
    description="Sistema avanzado de predicción de estadísticas NBA",
    python_requires=">=3.8",
) 