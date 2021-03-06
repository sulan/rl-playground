#! /usr/bin/env python

from __future__ import print_function
from gomoku.lib import gui

class ErrorHandler(object):
    def __enter__(self):
        pass

    def __exit__(self, error_type, error_args, traceback):
        if error_type is None:
            return
        if error_type is gui.tk.TclError:
            print("WARNING: A GUI-related `TclError` error occurred:", error_args)
            return True # do not reraise error
        # else:
        # GUIify errors
        if issubclass(error_type, Exception): # Only catch real exceptions not ``BaseException``
            gui.Message(message=repr(error_args), icon='warning', title='Gomoku - Error').show()
        return False # reraise nonTcl errors => stderr message and nonzero exit

if __name__ == '__main__':
    main_window = gui.MainWindow(16,16)
    with ErrorHandler():
        main_window.mainloop()
