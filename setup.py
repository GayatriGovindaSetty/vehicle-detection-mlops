from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vehicle-detection-mlops",
    version="1.0.0",
    author="Gayatri Govinda Setty",
    author_email="gayatrisetty27@gmail.com",
    description="Real-Time Vehicle Detection with MLOps Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vehicle-detection-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "ultralytics>=8.0.220",
        "torch>=2.1.0",
        "numpy>=1.24.3",
        "opencv-python>=4.8.1",
        "pillow>=10.1.0",
        "gradio>=4.8.0",
        "wandb>=0.16.1",
        "scikit-learn>=1.3.2",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
)