from setuptools import find_packages, setup
from typing import List

REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    with open(REQUIREMENT_FILE_NAME) as file_obj:
        requirement_list = file_obj.readlines()
    requirement_list = [req_name.replace("\n", "") for req_name in requirement_list]

    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)

    return requirement_list


setup(
    name="Medical Chatbot",
    version="0.0.1",
    author="Gaurav",
    author_email="gsr094@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
