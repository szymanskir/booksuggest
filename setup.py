from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Recommendation system for books. Bachelor thesis.',
    author='Paweł Rzepiński, Ryszard Szymański',
    license='',
    package_data={
        'models': ['*.pkl']    
    }
)
