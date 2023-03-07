FROM python:slim-buster
WORKDIR /app
COPY ./requirements.txt /app/
RUN pip3 install --upgrade pip \ 
	&& pip install --no-cache-dir  -r requirements.txt
COPY ./ /app
RUN python main.py
EXPOSE 80
