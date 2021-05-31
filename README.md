[![Python Version](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![GitHub repo size](https://img.shields.io/github/repo-size/DM1122/spotify-skip-prediction)](https://github.com/DM1122/spotify-skip-prediction)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/DM1122/spotify-skip-prediction)](https://github.com/DM1122/spotify-skip-prediction)

# Spotify Skip Prediction
APS360 project repository

# Contribution
## Setup
This section will take you through the procedure to configure your development environment. Ensure you have installed the project's python version.

The repo employs [poetry](https://python-poetry.org/) as its dependency and environment manager. Poetry can be installed through the Windows Powershell via:
```
$ (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

Clone the repo using github desktop or the commandline via:

```
$ git clone https://github.com/DM1122/spotify-skip-prediction.git
```

From within the cloned repo, run poetry's install command to install all the dependencies in one go:
```
$ poetry install
```

To make VSCode use the virtual environment that poetry created, add poetry's virtual environment path `C:\Users\<USERNAME>\AppData\Local\pypoetry\Cache\virtualenvs` to VSCode's `Venv Path` under `File>Preferences>Settings`. Once you have done so, enter the command pallet by going to `View>Command Palette` and search for `Python:Select Interpreter`. Select poetry's virtual environment for the repo.

You're now ready to start contributing!

## Commits
### Pre-Commit
This repo is configured to use [pre-commit](https://pre-commit.com/) hooks. The pre-commit pipeline is as follows:

1. [Isort](https://pycqa.github.io/isort/): Sorts imports, so you don't have to.
1. [Docformatter](https://github.com/myint/docformatter): Docstring formatter for those extra long docstrings.
1. [Black](https://black.readthedocs.io/en/stable/): The uncompromising code autoformatter.
1. [Flakehell](https://flakehell.readthedocs.io/): It's a wrapper for many linters.

Pre-commit will run the hooks on commit, but when a hook fails, they can be run manually to debug using:

```
$ isort . & docformatter . & black . & flakehell lint
```

### The 5 Rules of A Great Git Commit Message
<p align="center"><img src="https://imgs.xkcd.com/comics/git_commit.png" width="256"></p>

1. Write in the imperative
1. Capitalize first letter in the subject line 
1. Describe what was done and why, but not how
1. Limit subject line to 50 characters
1. End without a period

# Testing

This repo uses [pytest](https://docs.pytest.org/en/6.2.x/) for unit testing. To run unit tests, call:

```
$ pytest
```

You can find an interactive report of test results in `./logs/pytest-report.html`.