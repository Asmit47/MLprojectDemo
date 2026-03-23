from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","") for req in requirements]
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)    
    return requirements

setup(
    name="ml-project-demo",
    version="0.1.0",
    author="Asmit Kaushal",
    author_email="asmitkaushal47@gmail.com",
    description="A demo machine learning project",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    
)
