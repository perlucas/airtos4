FROM python:3.11.2

COPY . /usr/src/app

WORKDIR /usr/src/app/notebooks/airtos

RUN apt update
RUN apt install -y cmake libomp-dev

RUN pip install --upgrade pip
RUN pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
RUN pip install gymnasium "numpy<2.0" pandas pandas-ta matplotlib

CMD ["python", "run_sb3.py"]