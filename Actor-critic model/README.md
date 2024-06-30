[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/USeP6Ix8)
<br />
<p align="center">
  <h1 align="center">Reinforcement Learning - Final Project Template</h1>

  <p align="center">
  </p>
</p>

## About
This project contains skeleton code and a virtual environment to help you get started on the final project. You are free to add more packages and modules.
You are also free to change the project structure to use a convention you prefer. The one provided is the default project structure from `poetry`. 

## Getting started

### Prerequisites

- [GCC](https://gcc.gnu.org/) (a C++ compiler) - if you are using an environment that requires `box2d-py`.
- [Swig](https://swig.org/) - if you are using an environment that requires `box2d-py`.
- [Poetry](https://python-poetry.org/).

The first two dependencies are required for `box2d-py` because it is a dependency of `box2d-py` which some gymnasium environments use to render the environment.

## Running
<!--
-->

#### Setting up a virtual environment

You can also setup a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment (after `Add new interpreter`). The interpreter that Pycharm should use is `./.venv/bin/python3.10`.

If you want to add dependencies to the project then you can simply do
```
poetry add <package_name>
```

#### Running the docker container

Instead of running locally you can also run the program inside a container using docker. A `docker-compose.yaml` file is provided which you can use to run the container using `docker compose up --build`.

## Usage
You can add here some description on how to run the project (which file to run for example).

## Information on provided code

### Metrics Tracking
In the `util` package a singleton `MetricksTracker` class is provided. You can use it to keep track of average return and average loss values over time. Of course, you are also free to use other facilities for keeping track metrics.

### Function Approximators
Int he `models` package you can find PyTorch neural network classes for a standard multiLayer perceptron and a two-headed multilayer perceptron.

## Installing GCC and Swig

GCC stands for the GNU Compiler Collection and includes compilers for C and C++. To install GCC (assuming a Debian-based Linux distribution like Ubuntu):
```
apt-get install build-essential
```
For `swig`:
```
apt-get install swig
```
