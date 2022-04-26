from setuptools import setup

from central_tendency_rotations import __version__

setup(
    name="central_tendency_rotations",
    version=__version__,
    description="Computation of mean, median, and mode of a set of rotations (quaternions).",
    url="https://github.com/dilaragokay/central-tendency-rotations",
    author="Dilara Gokay",
    author_email="dilaragokay@gmail.com",
    py_modules=["central_tendency_rotations"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
    ],
)
