from setuptools import setup
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
	name = 'ScoreDraft',
	version = '1.0.3',
	description = 'A music/singing synthesizer that provides a Python based score authoring interface. ',
	long_description=long_description,
	long_description_content_type='text/markdown',  
	url='https://github.com/fynv/ScoreDraft',
	license='MIT',
	author='Fei Yang, Vulcan Eon, Beijing',
	author_email='hyangfeih@gmail.com',
	keywords='synthesizer audio music utau psola',
	packages=['ScoreDraft', "ScoreDraft.musicxml"],
	package_data = { 'ScoreDraft': ['*.dll', '*.so', '*.data']},
	install_requires = ['cffi', 'xsdata', 'python_ly', 'pyyaml'],
	entry_points={
        'console_scripts': [
            'scoredraft=ScoreDraft:run_yaml'
        ]
    }
)


