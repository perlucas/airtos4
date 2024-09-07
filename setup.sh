# !bin/sh

# Install Python 3.11
sudo apt update

sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz

tar -xvf Python-3.11.6.tgz

cd Python-3.11.6

sudo ./configure --enable-optimizations

sudo make -j 2

sudo make altinstall

python3.11 -V

sudo apt install python3-pip

pip3 -V

# Required dependencies
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg freeglut3-dev

# Python libraries
pip3 install tensorflow --upgrade
pip3 install scikit-image --upgrade
pip3 install tf-agents
pip3 install pyglet xvfbwrapper
pip3 install tf-keras
pip3 install keras-tuner
pip3 install pandas pandas-ta numpy

# Install Git
#sudo apt install git

#git --version

# Cloning the repo
#mkdir /content
#cd /content
#git clone https://github.com/perlucas/airtos4.git

cd /content/airtos4/notebooks/airtos/

echo "You're all set for the lab!"
echo "Run python3.11 execute_training.py to start training the model."