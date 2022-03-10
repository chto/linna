#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["numpy", "torch>=1.10.1", "emcee>=3.0.2", "h5py", "pydoe2", "scikit-learn", "torch-lr-finder", "matplotlib", "zeus-mcmc", "numdifftools", "sample_generator"]

test_requirements = ['pytest>=3', ]

setup(
    author="Chun-Hao To",
    author_email='chunhaoto@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    dependency_links=["git+https://github.com/tmcclintock/Training_Sample_Generator.git#egg=sample_generator"],
    description="Likelihood inference with neural network acceleration",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='linna',
    name='linna',
    packages=find_packages(include=['linna', 'linna.*', "script"]),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/chto/linna',
    version='0.0.2',
    zip_safe=False,
)
