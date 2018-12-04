#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

import versioneer


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', "openmm", "mdtraj"]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Andreas Krämer",
    author_email='kraemer.research@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Running and Analyzing OpenMM Jobs",
    entry_points={
        'console_scripts': [
            'rickflow=rickflow.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rickflow',
    name='rickflow',
    packages=find_packages(include=['rickflow']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Olllom/rickflow',
    zip_safe=False,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
