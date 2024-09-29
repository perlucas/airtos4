#! /bin/sh

sudo apt update
sudo apt install -y libhdf5-dev python3-h5py
sudo apt install -y python3-pip
sudo apt install -y git
sudo apt install -y python3-venv
git clone https://www.github.com/perlucas/airtos4.git ./airtos-dudqn
python3 -m venv ./airtos-dudqn
cd airtos-dudqn
source bin/activate
pip install tensorflow-probability==0.23.0 "tensorflow<=2.16" tf-keras tf-agents pandas pandas-ta numpy matplotlib scipy scikit-learn keras-tuner