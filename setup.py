from setuptools import setup

setup(
    name='SimulateRadarRainfall',
    url='https://github.com/amyycb/simradrain',
    author='Dr. Amy C. Green',
    author_email='amy.green3@newcastle.ac.uk',
    packages=['simradrain'],
    install_requires=['numpy', 'pandas', 'scipy', 'gstools', 'matplotlib', 'time', 'timeit', 'h5py', ],
    version='0.1.0',
    description='A Python package for extracting event properties from radar rainfall event data, and simulating '
                'realistic rainfall events based on parameters',
)
