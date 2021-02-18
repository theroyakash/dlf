from setuptools import setup, find_packages
from os import path

__version__ = "0.1"

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dlf',
    version=__version__,
    description='Easy Implementation for all popular deep learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/theroyakash/dlf',
    download_url='https://github.com/theroyakash/dlf/tarball/main',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=['numpy', 'torch'],
    dependency_links=dependency_links,
    author='theroyakash',
    author_email='hey@iamroyakash.com'
)