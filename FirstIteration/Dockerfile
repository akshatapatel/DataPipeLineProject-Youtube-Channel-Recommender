FROM python:3.6

COPY . /FirstIteration

RUN apt-get update
# RUN apt-get install -y mysql-server libmysqlclient-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r /FirstIteration/requirements.txt

ADD one_mill_trained_model.pkl /FirstIteration

CMD ["python", "/FirstIteration/app.py"]