"""
NICTO AI Setup Script
"""

from setuptools import setup, find_packages

setup(
    name="nicto-ai",
    version="0.1.0",
    description="NICTO AI - The World's Most Powerful Understanding Engine",
    author="NICTO AI Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "numpy>=1.26.0",
        "einops>=0.8.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.26.0",
        "datasets>=3.1.0",
        "accelerate>=1.1.0",
        "transformers>=4.46.0",
        "sentencepiece>=0.2.0",
        "tiktoken>=0.8.0",
        "wandb>=0.18.0",
        "tensorboard>=2.17.0",
        "rich>=13.9.0",
        "pydantic>=2.10.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=6.0",
            "ruff>=0.7.0",
        ],
    },
)
