FROM python:3.11.2

COPY . /usr/src/app

WORKDIR /usr/src/app/notebooks/airtos

RUN apt update
RUN apt install -y libhdf5-dev
# RUN pip install --upgrade "setuptools>=44.0.0"
# RUN pip install --upgrade "wheel>=0.37.1"
# RUN pip install --upgrade Cython
# RUN pip install --no-binary=h5py h5py
RUN apt install python3-h5py -y
RUN pip install tensorflow-probability==0.23.0 "tensorflow<=2.16" tf-keras tf-agents pandas pandas-ta numpy matplotlib scipy scikit-learn keras-tuner --upgrade

CMD ["python", "run_tunner.py"]