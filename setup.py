"""Setup script for CWCS package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cwcs",
    version="1.0.0",
    author="CWCS Team",
    description="Corpus-Wide Causal Scoring framework for Gene-Disease causal discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nency2/CWCS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "networkx>=2.8.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "neo4j": ["neo4j>=5.0.0"],
        "llm": [
            "openai>=1.0.0",
            "instructor>=0.5.0",
            "tiktoken>=0.5.0",
            "pydantic>=2.0.0",
            "tqdm>=4.65.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cwcs-pipeline=src.pipeline:main",
        ],
    },
)
