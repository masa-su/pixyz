import io
import os
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='pixyz',
    version=find_version("pixyz", "__init__.py"),
    packages=find_packages(),
    url='https://github.com/masa-su/pixyz',
    author='masa-su',
    author_email='masa@weblab.t.u-tokyo.ac.jp',
    description='Deep generative modeling library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch>=1.0",
        "torchvision",
        "tqdm",
        "scipy",
        "sympy>=1.4",
        "ipython",
        "tensorboardX",
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
    ],
)
