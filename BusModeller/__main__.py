from modules.MainMenu import MainMenu

def main():
    # Run main menu
    mainMenu = MainMenu()
    mainMenu.initializeMenu()

# If program is run as 'python BusModeller'
if __name__ == '__main__':
    main()
else:
    print("Please run the program by running the BusModeller folder with python. i.e. with the command 'python BusModeller'")