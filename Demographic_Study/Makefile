test:
	@find . -type d -name __pycache__ | xargs rm -rf
	@find . -type f -name "*.pyc" | xargs rm -rf
	python -m pytest tests

image:
	docker build -t 'demographic_data':$(git describe --tags)
