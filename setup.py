from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'localthickness',
    version = '0.1.2',
    description = 'Fast local thickness in 3D and 2D.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/vedranaa/local-thickness',
    author = 'VA Dahl and AB Dahl',
    author_email = 'vand@dtu.dk, abda@dtu.dk',
    license ='GPL-3.0 license',
    py_modules = ['localthickness'],
    install_requires = ['edt', 'numpy']
    )