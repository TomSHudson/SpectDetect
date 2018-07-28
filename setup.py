import setuptools

from os import path
from io import open
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="SpectDetect",
    version="0.0.3",
    author="Tom Hudson",
    description="Microseismic detection algorithm package based on using key features in the spectrum of a source to detect earthquakes over a given time period.",
    url="https://github.com/TomSHudson/SpectDetect",
    packages=['SpectDetect'],
    license='MIT',
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)
