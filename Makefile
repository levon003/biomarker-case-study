POETRY_HOME := ${HOME}/.local/bin
export PATH := ${POETRY_HOME}:$(PATH)

.PHONY: help install ensure-poetry install-poetry install-precommits install-kernel jupyter test

help:
	@echo "Relevant targets are 'install' and 'test'."

install:
	@$(MAKE) ensure-poetry
	@$(MAKE) install-precommits
	@$(MAKE) install-kernel

ensure-poetry:
	@if [ "$(shell which poetry)" = "" ]; then \
		$(MAKE) install-poetry; \
	else \
		echo "Found existing Poetry installation at $(shell which poetry)."; \
	fi
	@poetry install

install-poetry:
	@echo "Installing Poetry..."
	curl -sSL https://install.python-poetry.org | python3 -
	# TODO verify installation

install-precommits:
	@poetry run pre-commit autoupdate
	@poetry run pre-commit install --overwrite --install-hooks

install-kernel:
	@poetry run ipython kernel install --user --name=bcs

jupyter:
	@echo "Assuming Jupyter lab is installed and configured globally."
	@jupyter lab

test:
	@poetry run pytest --cov=src --cov-report term-missing
