from setuptools import setup, find_packages

setup(
    name='gamma_rena',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'pytest'
    ],
    author='Momsson',
    description='Gamma-RENA Project',
    license='MIT'
)