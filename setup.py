"""
Package setup
==================================

Author: Casokaks (https://github.com/Casokaks/)
Created on: Aug 15th 2021

"""


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='binclf',
    version='0.1.1',
    author='Casokaks',
    author_email='casokaks@gmail.com',
    description='ML binary classification utility package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Casokaks/binclf.git',
    project_urls = {
        "Bug Tracker": "https://github.com/Casokaks/binclf.git/issues"
    },
    license='MIT',
    packages=['binclf'],
    install_requires=[
        'sklearn', 
        'xgboost', 
        'seaborn', 
        'matplotlib', 
        'IPython', 
    ],  
)

