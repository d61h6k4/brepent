# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nox as a build tool."""

import nox

nox.options.sessions = ["lint", "tests", "typing"]


@nox.session(python=["3.9"],
             reuse_venv=True,
             venv_params=["--system-site-packages"])
def dev(session):
    """Create the dev environment.
    Fixed python version 3.9.
    Reuse venv: we don't need to have different
     clean environment for dev.
    Venv params: we need system site pacakges because
     pythonocc-core.
    """
    # This is dev environment, so preinstall neovim packages.
    session.install("neovim")
    # we use yapf for formating
    session.install("yapf")
    session.install("isort")
    # install the current package in editable mode
    session.install("-e", ".")
    message = "To activate the dev environment you can run the command:" \
              f"\n> source {session.virtualenv.bin}/activate"
    print(message)


@nox.session(python=["3.9"],
             venv_params=["--system-site-packages"])
def tests(session):
    """Run tests."""
    session.install("pytest")
    session.install("pytest-xdist")
    session.install(".")
    session.run("python", "-m", "pytest", "-n", "auto", "-rsx", "brepnet")


@nox.session(python=["3.9"],
             venv_params=["--system-site-packages"])
def lint(session):
    """Run linters."""
    def run_flake(source):
        return session.run("python", "-m", "flake8", "--count",
                           "--select=E9,F63,F7,F82,E225,E251", "--show-source",
                           "--statistics", source)

    def run_pylint(source):
        return session.run("python", "-m", "pylint", "--rcfile=.pylintrc",
                           source)

    session.install("flake8")
    session.install("pylint")

    for source in ["brepnet"]:
        run_flake(source)
        run_pylint(source)


@nox.session(python=["3.9"],
             venv_params=["--system-site-packages"])
def typing(session):
    """Run linters."""
    def run_pytype(source):
        return session.run("python", "-m", "pytype", "-k", "-d",
                           "import-error", source)

    session.install("pytype")
    session.install(".")

    for source in ["brepnet"]:
        run_pytype(source)
