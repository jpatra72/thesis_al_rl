from setuptools import setup

setup(
    name='custom_envs',
    version='0.0.1',
    packages=['custom_envs'],
    install_requires=['gymnasium'],
    description='Custom Environments for Active Learning using Reinforcement Learning',
    author='Jyotirmaya Patra',
    options={'clean': {'all': True}}
)