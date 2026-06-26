test:
	pytest tests/ -v

lint:
	pylint model.py predict.py --disable=C0114,C0115,C0116

format:
	black model.py predict.py tests/
	isort model.py predict.py tests/

quality: format lint test
