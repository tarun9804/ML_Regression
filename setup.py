from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return
    :param file_path:
    :return:
    """
    requirements = []
    with open(file_path) as file_obj:
        req = file_obj.read()
        requirements = req.split('\n')
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Tarun',
    author_email='tarun9804@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
