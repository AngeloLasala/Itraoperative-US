from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='intraoperative_us',
    version='0.0',
    description='Diffusion model for neurosurgery iUS',
    author='Angelo Lasala',
    author_email='Lasala.Angelo@santannapisa.it',
    packages=find_packages(),
    install_requires=requirements,
)