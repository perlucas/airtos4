FROM python:3.11.2

COPY . /usr/src/app

WORKDIR /usr/src/app/notebooks/airtos

RUN apt update
RUN apt install -y libhdf5-dev
RUN apt install python3-h5py -y

# install tensorflow
RUN pip install tensorflow-probability==0.23.0 "tensorflow<=2.16" tf-keras tf-agents pandas pandas-ta numpy matplotlib scipy scikit-learn keras-tuner --upgrade


CMD [ "python", "./a3c_demo.py" ]
