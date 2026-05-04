from setuptools import find_packages, setup

setup(
    name="data_projects_hackathon",
    version="0.1.0",
    description="Utilities and preprocessing code for the Hackathon 3 project.",
    author="Alessia, Tommaso, Giacomo",
    author_email="",
    python_requires=">=3.11",
    packages=find_packages(include=["utils", "utils.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pyarrow",
        "tqdm",
        "networkx",
        "xgboost",
        "sentence-transformers",
        "transformers",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)