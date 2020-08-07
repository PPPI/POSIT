#!/usr/bin/env python

from setuptools import setup

setup(name='POSIT',
      version='0.3.5',
      description='Part-of-Speech tagger for English and Source-code',
      author='Profir-Petru Partachi',
      author_email='p.partachi@cs.ucl.ac.uk',
      packages=['tagger', 'preprocessor', 'baseline', 'StORMeD'],
      package_dir={
            'tagger': 'src/tagger',
            'preprocessor': 'src/preprocessor',
            'baseline': 'src/baseline',
            'StORMeD': 'src/baseline/StORMeD',
            },
      package_data={'tagger': ['data/*.pkl']},
      setup_requires=[
            'tensorflow', 
            'numpy', 
            'gensim', 
            'nltk', 
            'xmltodict',
            'beautifulsoup4', 
            'html5lib', 
            'scikit-learn',
            'tqdm',
            'json_lines',
            'antlr4',
            'psutil',
            'joblib',
            'h5py',
      ], install_requires=['snorkel', 'pandas']
      )