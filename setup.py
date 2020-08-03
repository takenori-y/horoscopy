#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import find_packages, setup


exec(open('horoscopy/version.py').read())

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='horoscopy',
    version=__version__,
    description='Python module for speech signal processing',
    author='Takenori Yoshimura',
    author_email='takenori@sp.nitech.ac.jp',
    url='https://github.com/takenori-y/horoscopy',
    download_url='',
    packages=find_packages(exclude=('docs', 'tests')),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='speech signal dsp',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>= 3.6',
    install_requires=[
        'numpy >= 1.15.0',
        'scipy >= 1.4.0',
        'librosa >= 0.8.0',
    ],
    extras_require={
        'dev': [
            'flake8',
            'numpydoc',
            'pytest',
            'sphinx',
            'sphinx_rtd_theme',
            'twine',
        ],
    },
)
