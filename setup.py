from setuptools import find_packages, setup


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='horoscopy',
    version='0.1.0',
    description='Python module for speech signal processing',
    author='Takenori Yoshimura',
    author_email='takenori@sp.nitech.ac.jp',
    url='https://github.com/takenori-y/horoscopy',
    download_url='',
    packages=find_packages(exclude=('tests')),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='speech signal dsp',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6',
    install_requires=[
        'librosa >= 0.8.0',
        'numpy >= 1.15.0',
        'scipy >= 1.4.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'sphinx',
            'sphinx_rtd_theme',
            'numpydoc',
        ],
    },
)
