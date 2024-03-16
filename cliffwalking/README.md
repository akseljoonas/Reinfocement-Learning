<br />
<p align="center">
  <h1 align="center">Reinforcement Learning</h1>

  <p align="center">
  </p>
</p>

## About
This project contains skeleton code and a virtual environment to help you get started on the programming of reinforcement learning agents.
The project contains one package `cliffwalksim` in which you can add your code. You are free to add more packages and modules but for the programming of the simulation that should not be necessary.
You are also free to change the project structure to use a convention you prefer. The one provided is the default project structure from `poetry`. 

## Getting started

### Prerequisites

- [GCC](https://gcc.gnu.org/) (a C++ compiler).
- [Swig](https://swig.org/).
- [Poetry](https://python-poetry.org/).

The first two dependencies are required because it is a dependency of `box2d-py` which some gymnasium environments use to render the environment.

## Installing GCC and Swig

GCC stands for the GNU Compiler Collection and includes compilers for C and C++. To install GCC (assuming a Debian-based Linux distribution like Ubuntu):
```
apt-get install build-essential
```
For `swig`:
```
apt-get install swig
```

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

### Agent Creation
You can add RL agents by inheriting from the `TabularAgent` abstract base class and adjusting the `AgentFactory` class.

### Metrics Tracking

`MetricsTracker` is an object that you can use to record metrics of the reinforcement learning agents. Currently, it provides functionality for keeping track of the sample mean and variance of the return over time (per episode) for each agent. This can naturally be extended to record the mean and variance of any value over time. There is also a `plot` function you can use to plot the saved data.

`MetricsTracker` can be used as follows: Suppose an agent with name `agent_id` received `return` after an episode has finished, then you can record the return as follows:
```
tracker = MetricksTracker()
tracker.record_return(agent_id, return)
```
At the end you can plot the results using:
```
tracker.plot()
```
which will give a plot showing the average return over time (with variance) for each agent.

You are free to modify `MetricksTracker` to suit your needs.