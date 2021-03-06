[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python Version](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![GitHub repo size](https://img.shields.io/github/repo-size/DM1122/spotify-skip-prediction)](https://github.com/DM1122/spotify-skip-prediction)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/DM1122/spotify-skip-prediction)](https://github.com/DM1122/spotify-skip-prediction)
![Lines of code](https://img.shields.io/tokei/lines/github/DM1122/fpcnn)

# Spotify Skip Prediction
<p align="left"><img src="img/spotify-logo.png" height="64"></p>

**APS360 Final Project**

The challenge: predict if users will skip or listen to the music they're served on the Spotify music streaming platform. A solution to the Spotify Sequential Skip Prediction Challenge 2018 on [AIcrowd](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge). A supervised sequential classification task.

Recommender systems play a crucial role in the digital age where media is abundant, and serving the right media to the right user at the right time means everything in terms of revenue generation. It is in the best interest of digital media providers the likes of Spotify, Youtube, Facebook, Google, and others to maximize audience retention. A model which is able to accurately predict user intent given significant nuance is necessary; machine learning is consequently the right tool for the task.

# System Architecture
<p align="center"><img src="img/system-diagram.png" height="256"></p>

We employ an autoencoder model for feature extraction and a vanilla RNN model for inference on the autoencoder embeddings. Our results are evaluated against a gradient boosting model for classification.

# Contribution
## Setup
This section will take you through the procedure to configure your development environment. At a glance:
1. Install project's python version
1. Install Git
1. Install poetry
1. Clone repository
1. Run poetry install
1. Configure IDE virtual environment
1. Install pre-commit hooks

Begin by installing the project's python version. See the badges at the top of the README for the version number.

If not already installed, install [git](https://git-scm.com/).

The repo employs [poetry](https://python-poetry.org/) <img src="img/poetry-logo.png" height="16"/> as its dependency and environment manager. Poetry can be installed through the Windows Powershell via:
```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py -UseBasicParsing).Content | python -
```

Clone the repo using [Github Desktop](https://desktop.github.com/) <img src="img/github-desktop-logo.png" height="16"/> or the commandline via:

```
git clone https://github.com/DM1122/spotify-skip-prediction.git
```

From within the cloned repo, run poetry's install command to install all the dependencies in one go:
```
poetry install
```

Configure your IDE to use the virtual environment poetry has created at `C:\Users\<USERNAME>\AppData\Local\pypoetry\Cache\virtualenvs`. In the case of [VSCode](https://code.visualstudio.com/) <img src="img/vscode-logo.png" height="16"/>, enter the command pallet by going to `View>Command Palette` and search for `Python:Select Interpreter`. Select the appropriate poetry virtual environment for the repo.

Install the pre-commit script and hooks using:
```
pre-commit install --install-hooks
```

You're now ready to start contributing!

## Adding Packages
To add a new package to the poetry virtual environment, install it via:
```
poetry add <package>
```
This is poetry's version of `pip install <package>`.

## Testing
This repo uses [pytest](https://docs.pytest.org/en/6.2.x/) for unit testing. To run all unit tests, call:

```
pytest -v
```

You can find an interactive report of test results in `./logs/pytest-report.html`. Individual tests can also be specified as follows:
```
pytest tests/test_<filename>.py::<function name>
```

Groups of tests can be run using markers. Assign a marker decorator to the group of functions you want to test like this:

```
@pytest.mark.foo
def my_test_function():
    # some test
```

To use the custom marker `foo`, it must be added to the list of custom pytest markers in `pyproject.toml>[tool.pytest.ini_options]>markers`. The tests marked with `foo` can then be run by calling:
```
pytest -v -m foo
```

Or to avoid all tests with a particular marker, call:
```
pytest -v -m "not foo"
```


## Commits
### Pre-Commit
This repo is configured to use [pre-commit](https://pre-commit.com/) hooks. The pre-commit pipeline is as follows:

1. [Isort](https://pycqa.github.io/isort/): Sorts imports, so you don't have to.
1. [Black](https://black.readthedocs.io/en/stable/): The uncompromising code autoformatter.
1. [Pylint](https://github.com/pycqa/pylint): It's not just a linter that annoys you!

Pre-commit will run the hooks on commit, but when a hook fails, they can be run manually to debug using:

```
isort . ; black . ; pylint_runner
```



