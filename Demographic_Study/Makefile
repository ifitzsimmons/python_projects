test:
	@find . -type d -name __pycache__ | xargs rm -rf
	@find . -type f -name "*.pyc" | xargs rm -rf
	python -m pytest tests
