#!/usr/bin/env bash

uv run ruff check --select I --fix
uv run ruff check --fix
uv run ruff format src tests
