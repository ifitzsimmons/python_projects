FROM python:3.6.4-slim
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["python", "find_similar_states.py"]
