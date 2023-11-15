import os
import subprocess
from configparser import ConfigParser
from ast import literal_eval

# Check if setup is complete
config = ConfigParser()
config.read('configs/config.ini')

if literal_eval(config['General']['setup']) == False:
    print("Program is not set up. Run setup.py to set up the program")
    exit()

# Run program through virtual environment
try:
    subprocess.run('venv/Scripts/python.exe BusModeller')
except subprocess.CalledProcessError as e:
    print("Error: ", e, "\nIf this occurs again, try to setup the program again using setup.py, or manually set it up using the steps in README.md")