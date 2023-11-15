import tkinter as tk
from tkinter import messagebox

# Class is designed just to handle hierarchy of program windows, nothing else
# When the Menu's window is closed using the windows 'X' button in the top right, all descendants (children of children) will also be closed recursively
# parent should be another Menu object, and children should be a list of Menu objects
class MenuNode():

    def __init__(self, parent=None, children=None):

        if parent == None:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel(master=parent.window)

        # Calls 'close' method when the close button is pressed
        # In 'close' method, all descendants are also iteratively closed, and self is removed from parent's children
        self.window.protocol('WM_DELETE_WINDOW', self.close)

        # If this window has a parent, add this window to the parent's children
        if parent != None:
            parent.children.append(self)
        self.parent = parent
        
        # Should normally be None
        if children == None:
            self.children = []
        else:
            self.children = children


    # If a parent closes, all children (as well as further descendants) should close too
    # When ask is True, a prompt is shown to the user if they want to close all descendant windows too
        # When close is called for the descendants, ask is set to False
    def close(self, ask=True):

        # If self has any children, recursively close all the descendants of self
        if len(self.children) != 0:

            # Prompt to ask user if they're ok with closing the entire program
            if messagebox.askokcancel('Quit', 'Closing this menu will also close all descendent menus. Is this ok?') or ask == False:

                # Iteratively close all descendant menus
                # If a copy of the children isn't fixed, problems arise from the self.parent.children.remove(self) step - some children end up not being deleted
                temp = self.children.copy()
                for c in temp:
                    c.close(ask=False)

            else:
                return

        # Remove window from parent's children
        if self.parent != None:
            self.parent.children.remove(self)

        # Destroy own window
        self.window.destroy()