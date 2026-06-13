# Install from scratch

This guide sets up everything needed to run the project on a fresh machine,
starting from nothing. It covers macOS and Windows. By the end you will have
the right Python version, the Poetry dependency manager, and all project
dependencies installed.

There are four steps:

1. [Make sure Python 3.12 is installed](#step-1-install-python-312)
2. [Install pipx](#step-2-install-pipx)
3. [Install Poetry](#step-3-install-poetry-with-pipx)
4. [Install the project](#step-4-install-the-project)

If you already have a working Poetry and a Python 3.12 or 3.13 interpreter,
skip straight to [step 4](#step-4-install-the-project).

## Why these versions

The project requires Python **3.12 or 3.13** (`>=3.12,<3.14`). The upper
bound matters: PyTorch does not yet publish installable packages ("wheels")
for Python 3.14, so a 3.14 interpreter will fail to install the project even
though it is newer. If your computer came with Python 3.14, that is fine —
you do not have to remove it. You just need a 3.12 installed alongside it,
and the steps below show how to point Poetry at the right one.

3.12 is recommended so every machine on the project matches.

## Step 1: Install Python 3.12

First check what you already have. Open a terminal and run:

```sh
python3.12 --version
```

If that prints `Python 3.12.x`, you already have it — skip to
[step 2](#step-2-install-pipx). If instead you see "command not found" (macOS)
or the command is not recognized (Windows), install it as follows.

### macOS

Install with [Homebrew](https://brew.sh/). If you do not have Homebrew, install
it first with the one-line command on its homepage, then:

```sh
brew install python@3.12
```

Confirm it worked:

```sh
python3.12 --version
```

This installs 3.12 next to any other Python you have; it does not replace your
system Python or any existing version.

### Windows

Download the **Python 3.12** installer from
[python.org/downloads](https://www.python.org/downloads/) (pick the latest
3.12.x release, not 3.13 or 3.14) and run it. On the first screen of the
installer, check **"Add python.exe to PATH"** before clicking Install.

Then open a **new** terminal (PowerShell) and confirm:

```powershell
py -3.12 --version
```

## Step 2: Install pipx

[pipx](https://pipx.pypa.io/) installs Python command-line applications (like
Poetry) into isolated environments, so they do not interfere with your
projects or with each other. It is the recommended way to install Poetry.

### macOS

```sh
brew install pipx
pipx ensurepath
```

`pipx ensurepath` adds pipx's install location to your `PATH`. After running
it, **close the terminal and open a new one** so the change takes effect.

### Windows

```powershell
py -3.12 -m pip install --user pipx
py -3.12 -m pipx ensurepath
```

Then **close the terminal and open a new one** so the updated `PATH` is
picked up.

Confirm pipx is available:

```sh
pipx --version
```

## Step 3: Install Poetry with pipx

[Poetry](https://python-poetry.org/) is the tool this project uses to manage
its virtual environment and dependencies.

```sh
pipx install poetry
```

Confirm it worked:

```sh
poetry --version
```

This is the only thing you install globally; everything the project itself
needs is installed in step 4, into a virtual environment that Poetry manages.

## Step 4: Install the project

From the repository root (the folder containing `pyproject.toml`), point
Poetry at your 3.12 interpreter, then install:

### macOS

```sh
poetry env use python3.12
poetry install
```

### Windows

Find the full path to your 3.12 interpreter, then hand it to Poetry:

```powershell
py -3.12 -c "import sys; print(sys.executable)"
poetry env use "C:\path\printed\above\python.exe"
poetry install
```

Passing the explicit path is the most reliable approach on Windows, since
there is usually no plain `python3.12` command for Poetry to find.

`poetry env use` tells Poetry which Python to build the project's virtual
environment with. This is the step that avoids accidentally using Python 3.14.
You only need to run it once per machine; afterwards `poetry install` reuses
the same environment.

`poetry install` creates the virtual environment and installs every
dependency, including PyTorch and ultralytics. The first run downloads several
hundred megabytes and can take a few minutes. On Windows and Linux the CUDA
build of PyTorch is installed; on macOS the standard build is installed, which
supports the Apple Silicon GPU through MPS (see
[Choose a device](choose-a-device.md)).

## Verify the installation

Confirm the core libraries import correctly:

```sh
poetry run python -c "import torch, ultralytics; print(torch.__version__)"
```

This should print a torch version such as `2.7.1` (Windows/Linux show
`2.7.1+cu128`). On an Apple Silicon Mac you can also confirm the GPU backend
is available:

```sh
poetry run python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

For a full end-to-end check of the actual pipeline, continue to
[Run a smoke test](run-a-smoke-test.md).

## Troubleshooting

**`Unable to find installation candidates for torch` / wheels "skipped" on
ABI tags.** Poetry is using a Python version with no PyTorch wheels, almost
always 3.14. Check which interpreter the environment uses with
`poetry env info`. If it reports 3.14, run `poetry env use python3.12`
(macOS) or `poetry env use py -3.12` (Windows) and `poetry install` again.

**`poetry: command not found` after step 3.** Your `PATH` was not refreshed.
Run `pipx ensurepath` again and open a new terminal window.

**`python3.12: command not found` in step 4 on macOS.** Use the full path
Homebrew prints, for example
`poetry env use /opt/homebrew/opt/python@3.12/bin/python3.12`.
