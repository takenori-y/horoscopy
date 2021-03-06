# Copyright (c) 2020 Takenori Yoshimura
# Licensed under the MIT license

.PHONY: init dev test wheel clean

init:
	pip install -e .

dev:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate; pip install -e .[dev]
	touch venv/bin/activate

test:
	. venv/bin/activate; pytest -s

wheel:
	rm -rf dist
	. venv/bin/activate; python setup.py bdist_wheel

clean:
	rm -rf build dist docs/_build venv *.egg-info
	find . -name '__pycache__' -type d | xargs rm -rf
