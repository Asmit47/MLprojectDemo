from setuptools import setup, find_packages

setup(
    name="ml-project-demo",
    version="0.1.0",
    author="Asmit Kaushal",
    description="A demo machine learning project",
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
)
