# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.0.1",
    packages=find_packages(),
    author="Yannik Rath",
    description="Implementation of the transformer architecture from 'Attention is All You Need'.",
    install_requires=["torch", "torchtext"],
)
