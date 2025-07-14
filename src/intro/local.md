---
layout: post
title:  2. Local Setup (optional)
parent: Intro
nav_order: 3
---

## Local Setup
If you'd like, you can set up a working environment on your own computer. If you would like to use [Google Colab](https://colab.research.google.com) instead, then you can skip this article.

## IDE ([vscode](https://code.visualstudio.com) recommended)

To follow along with this course you'll need an Integrated Development Environment (IDE). This is where you'll be able to write and run all of your code. If you don't have one already, we recommend installing [VSCode](https://code.visualstudio.com).

## Python

Once you have your IDE installed, you'll need to install python on your system. [Python's site](https://www.python.org/downloads/) should help you download and setup python on your system.

## pip vs. conda

Python has many **package managers** which help you download and use libraries from the web, such as numpy, pandas, and many others which we will cover throughout the course.

### pip
If you are using **Windows**, I'd recommend using ```pip```, which comes preinstalled with Python.
To create a new environment with pip:
1. ```python -m venv env_name```
2. ```.\env_name\Scripts\activate```

To install packages in your environment, run ```pip install <package_name>```.
To deactivate your environment, run ```deactivate```

### conda
If you are using **MacOS**, ```Anaconda``` is a very usefull tool for managing installs and environments. It is available on Windows, but I found it to be tough to use as it was not enabled in all terminals. To install Conda:
1. Install [Homebrew](https://brew.sh) (optional, but recommended)
2. Run the command ```brew install --cask anaconda``` to install conda on your system.

To create a new environment, you can use ```conda create -n env_name python=3.9```. 
To activate your environment, run ```conda activate env_name```.
To install packages in your environment, use ```conda install <package_name>```. 
To deactivate your enviuronment, run ```conda deactivate```.

## git

Git is an important way to interact with your files. It is a version control system and is commonly used with [Github](https://github.com), a large code storage, sharing, and hosting site. If you don't already have a Github account, [create one now](https://github.com/signup).
To get started with git, install and setup using [their site](https://git-scm.com/downloads).

VSCode has a built-in version control tab that allows you to interact with git and github. For more info, visit [their guide](https://code.visualstudio.com/docs/sourcecontrol/overview).