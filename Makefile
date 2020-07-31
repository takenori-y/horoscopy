.PHONY: init dev test clean

init:
	pip install -e .

dev:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate; pip install -e .[dev]
	touch venv/bin/activate

test:
	. venv/bin/activate; pytest -s

clean:
	rm -rf venv
