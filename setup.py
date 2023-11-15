import os
import venv
import subprocess
from ast import literal_eval
import sys
from configparser import ConfigParser

# Initialize ConfigParser
# config = ConfigParser()
# config.read('configs/setup')
setup = open('configs/setup.txt','r').read()

# If setup variable is not true, set program up
# if literal_eval(config['General']['setup']) == False:
if setup == 'False':

    print("Setting up the program. Please don't interrupt this process - it may take a few minutes")

    # Create the virtual environment
    if os.path.exists('venv') == False:
        venv.create(env_dir='venv', with_pip=True)

    try:

        # Install dependencies
        # Commands to run venv are slightly different for UNIX-based and Windows systems
        if sys.platform == 'win32':
            subprocess.run('venv\\scripts\\pip.exe install -r requirements.txt', shell=True, check=True)
            print("Dependencies successfully installed")
        else:
            subprocess.run('source venv\\scripts\\activate && pip install -r requirements.txt', shell=True, check=True)
            # subprocess.run('.venv')

    except subprocess.CalledProcessError as e:
        print("There has been some error which prevents the requirements from being installed to your system.")
        print("Please try to manually enter the virtual environment .venv and manually install the requirements in requirements.txt using pip, then try run the program using 'python run.py'")
        exit()

    except Exception as e:
        print("Unexpected error in 'setup.py': ", e)
        exit()

    # config.set('General', 'setup', 'True')
    # with open('configs/config.ini', 'w') as configFile:
    #     config.write(configFile)
    with open('configs/setup.txt', 'w') as s:
        s.write('True')

else:
    print("Program has already been set up")
    print("If you need to set up the program again, delete the 'venv' folder, then change the text in 'configs/setup.txt' to 'False', then run this script again")
    exit()