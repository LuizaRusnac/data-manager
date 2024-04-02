from setuptools import setup, find_packages

# with open('requirements.txt') as f:
#     requirements = f.read()

# # requirements = [line for line in requirements]
# print(requirements)
# input("pause")

setup(
    name='data-manager',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
        ]
    }
)