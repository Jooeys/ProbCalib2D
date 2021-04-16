import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="probability-calibration",
    version="0.0.1",
    author="Junyi Zhong",
    author_email="jyzh@yahoo.com",
    description="Utilities to calibrate model outcome probability and evaluate calibration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jooeys/ProbCalib2D",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'sklearn', 'parameterized'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)