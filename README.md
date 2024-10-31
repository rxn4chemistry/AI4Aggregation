# ai4agg

## Add CI badges
Add the CI badges by adding the following line to the README.md: 

## Scan Open Source Software Usage
Generate a requirements.txt file that can be scanned for whitesourcing with Mend. 
```console
poetry export -f requirements.txt
```
Activate the scanning. 

## Development setup
To install the package run:

```console
poetry install
```

## Using Ruff

```console
poetry run ruff check .          # Lint all files in the current directory.
poetry run ruff check . --fix    # Lint all files in the current directory, and fix any fixable errors.
poetry run ruff check . --watch  # Lint all files in the current directory, and re-lint on change.
```
