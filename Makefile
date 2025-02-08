venv: venv/bin/activate

venv/bin/activate: poetry.lock
	python3 -m venv ./venv
	. venv/bin/activate && pip install poetry && poetry install

.PHONY: package
package: venv
	rm -fr dist/*
	venv/bin/python -m build

.PHONY: deploy-to-pypi
deploy-to-pypi: package
	venv/bin/twine upload dist/*

.PHONY: clean
clean:
	rm -fr dist/*
	rm -fr venv
