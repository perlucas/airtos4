# Airtos v4

## Setup a Droplet

1. Run `sudo apt update` and `sudo apt install -y cmake libomp-dev`
2. Install Git: `sudo apt install -y git`
3. Clone the repo: `git clone https://www.github.com/perlucas/airtos4.git`
4. Make sure Python 3 is installed: `python3 -V`. Install pip: `sudo apt install python3-pip -y`
5. Install venv: `sudo apt install -y python3.11-venv`. Setup a virtual env in the cloned folder: `python3 -m venv airtos4`
6. Activate environment (`source bin/activate`) and install dependencies:
```
pip install --upgrade pip
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
pip install gymnasium "numpy<2.0" pandas pandas-ta matplotlib "tensorflow<=2.16" scipy scikit-learn optuna
pip install sb3-contrib
```
7. Do not forget to switch to `sb3` branch
