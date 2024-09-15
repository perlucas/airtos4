FROM python:3.11

# install hpy5 (required by tensorflow)
RUN apt-get update
RUN apt install -y libhdf5-dev
RUN pip install --upgrade "setuptools>=44.0.0"
RUN pip install --upgrade "wheel>=0.37.1"
RUN pip install --upgrade Cython
RUN pip install --no-binary=h5py h5py

# install tensorflow
RUN pip install tf-agents tf-keras "tensorflow==2.15" numpy pandas pandas-ta matplotlib

COPY . /usr/src/app

WORKDIR /usr/src/app/notebooks/airtos

CMD [ "python", "./a3c_demo.py" ]
