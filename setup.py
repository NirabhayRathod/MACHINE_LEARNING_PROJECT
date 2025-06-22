from setuptools import find_packages, setup
from typing import List

c='-e .'
def function(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[i.replace('\n','') for i in requirements]
        
        if c in requirements:
            requirements.remove(c)
    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Nirabhay',
    author_email='nirbhay105633016@gmail.com',
    packages=find_packages(),
    install_requires=function('requirements.txt')
)