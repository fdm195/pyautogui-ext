import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyautogui-ext",
    version="0.0.1",
    author="Weihong Xu",
    author_email="xuweihong.cn@gmail.com",
    description="helpful extension for pyautogui",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/link89/pyautogui-ext",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'fire',
        'pyscreeze',
        'opencv-python-headless',
        'opencv-contrib-python-headless',
        'numpy',
        'matplotlib',
    ],
)
