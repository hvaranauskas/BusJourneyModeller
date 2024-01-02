# Program setup:
Generate an API key for collecting bus data by creating an account on https://data.bus-data.dft.gov.uk/account/login/ then finding your key in 'Account Settings'\
Navigate to /configs/defaults.ini then enter your API key in the 'apikey' row

Need Python 2.7+\

Automatic setup by running 'setup.py' using your python.exe installation:
- This will do the same steps as in the manual setup below (including setting up a virtual environment)

Manual setup:
- Optionally create a virtual environment for the application
- Install requirements from 'requirements.txt' using pip
- Change '/configs/setup.txt' from 'True' to 'False'

NOTE: The requirements take up ~500MB of storage

# Running the program:
If you set up the program, using setup.py, you can run the program by running run.py\
You can also use the command 'python.exe BusModeller' to run the program, if your python.exe has the requirements installed

## /configs/defaults.ini:
Stores default values for the entries in the UI menu\
Changing an entry in the UI should update the default entry in defaults.ini

# Uninstalling:
If dependencies are installed in a virtual environment, they will be removed simply by deleted the 'Application' folder\
Otherwise, you'll have to manually uninstall the dependencies\
If you changed a data directory in configs to store data in a directory outside the project, you will have to manually delete this data
