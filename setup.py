from setuptools import setup, find_packages

setup(
    name='pd-data-manager',
    version='0.1.1',
    packages=find_packages(),
    author='Luiza Rusnac',
    author_email='luiza.rusnac93@gmail.com',
    description='A package for efortless data manager.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LuizaRusnac/data-manager.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)