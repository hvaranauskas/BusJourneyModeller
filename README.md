**Program setup:**
You need Python 2.7+ to use this program
Generate an API key for collecting bus data by creating an account on https://data.bus-data.dft.gov.uk/account/login/ then finding your key in 'Account Settings'
Navigate to /configs/defaults.ini then enter your API key in the 'apikey' row

Recommended to run 'python.exe setup.py', which will set up a virtual environment called 'venv' in the root folder 'Application' and install the requirements to this virtual environment.
You could also run 'python.exe -m pip install -r requirements.txt', but you will have to manually uninstall the dependencies once you are done with them
NOTE: The requirements take up ~500MB of storage

Running the program:
	If you set up the program, using setup.py, you can run the program by running run.py
	You can also use the command 'python.exe BusModeller' to run the program, if your python.exe has the requirements installed

/configs/defaults.ini:
	Stores default values for the entries in the UI menu
	Changing an entry in the UI should update the default entry in defaults.ini

Uninstalling:
	If dependencies are installed in a virtual environment, they will be removed simply by deleted the 'Application' folder
	Otherwise, you'll have to manually uninstall the dependencies
	If you changed a data directory in configs to store data in a directory outside the project, you will have to manually delete this data
