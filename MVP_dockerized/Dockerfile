FROM python:3.6

COPY . /MVP_dockerized

RUN apt-get update
# RUN apt-get install -y mysql-server libmysqlclient-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r /MVP_dockerized/requirements.txt

ADD one_mill_trained_model.pkl /MVP_dockerized

CMD ["python", "/MVP_dockerized/app.py"]