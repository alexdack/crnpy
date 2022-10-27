from setuptools import find_packages, setup
setup(
    name='crnpy',
    packages=find_packages(include=['crnpy']),
    version='0.1.6',
    description='A library to simulate chemical reaction networks.',
    author='Alex Dack',
    license='MIT',
    install_requires=['numpy', 'scipy', 'matplotlib'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)