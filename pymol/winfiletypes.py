r'''
Register file extensions under Microsoft Windows.

https://stackoverflow.com/questions/16829736/windows-changing-the-name-icon-of-an-application-associated-with-a-file-type
https://msdn.microsoft.com/en-us/library/windows/desktop/ee872121(v=vs.85).aspx
https://msdn.microsoft.com/en-us/library/windows/desktop/hh127445(v=vs.85).aspx
https://msdn.microsoft.com/en-us/library/windows/desktop/cc144148(v=vs.85).aspx


HKEY_CLASSES_ROOT
   PyMOL.pse   <----------------------------------- ProgId
      FriendlyTypeName = PyMOL Session File <------ plain text works on Win10!
      DefaultIcon
         (Default) = C:\...\pymolfile.ico
      shell
         open
            (Default) = Open with PyMOL
            command
               (Default) = C:\...\pymol.exe "%1"
         play
            (Default) = Present with PyMOL
            command
               (Default) = C:\...\pymol.exe -A6 "%1"
   PyMOL.pml   <----------------------------------- ProgId
      PerceivedType = text <----------------------- Adds "Notepad" to "Open With"
      FriendlyTypeName = PyMOL Script <------------ works on Win10
      DefaultIcon
         (Default) = C:\...\pymolfile.ico
      shell
         open
            (Default) = Run with PyMOL
            command
               (Default) = C:\...\pymol.exe "%1"
   .pse
      (Default) = PyMOL.pse
   .pze
      (Default) = PyMOL.pse
   .pml
      (Default) = PyMOL.pml

HKEY_CURRENT_USER
   Software
      Schrodinger
         PyMOL
            PYMOL_PATH   <------------------------- Checked by "Send to PyMOL..."
      Microsoft
         Windows
            CurrentVersion
               Explorer
                  FileExts
                     .pse
                        OpenWithList
                        OpenWithProgIds  <--------- auto added
                           PyMOL.pse = "" <-------- remove
                        UserChoice    <------------ remove
                           Progid = PyMOL.pse

HKEY_CLASSES_ROOT == combined(
    HKEY_CURRENT_USER\Software\Classes
    HKEY_LOCAL_MACHINE\Software\Classes
)


'''

import os
import sys
import ctypes

import winreg

descriptions = {
    'pdb': 'Protein Data Bank File',
    'cif': 'Crystallographic Information File',
    'mmtf': 'Macromolecular Transmission File',
    'pml': 'PyMOL Script',
    'pse': 'PyMOL Session File',
    'psw': 'PyMOL Show File',
    'pwg': 'PyMOL Web GUI',
    'py': 'Python Script',
    'r3d': 'Raster3D File',
    'mae': 'Maestro Structure File',
    'moe': 'MOE Structure File',
    'ph4': 'MOE Pharmacophore File',
    'cms': 'Desmond Structure File',
    'idx': 'Desmond Trajectory',
    'ccp4': 'CCP4 Map File',
    'acnt': 'Sybyl Map File',
    'gro': 'Gromacs Structure File',
    'brix': 'BRIX Density Map',
    'dx': 'DX Potential Map',
    'mtz': 'MTZ Reflection File',
    'spi': 'Spider Map File',
    'vdb': 'VIPERdb Structure File',
    'pdbqt': 'AutoDock Structure File',
    'grd': 'Delphi Potential Map',
    'phi': 'Delphi Potential Map',
    'xplor': 'XPLOR Electron Density Map',
}

texttypes = set(['pml', 'py'])

# boolean value is True if registration of the extension is recommended
extensions = {
    'acnt': False,
    'brix': False,
    'cc1': False,
    'cc2': False,
    'ccp4': True,
    'cif': True,
    'cms': False,
    'cube': False,
    'dsn6': False,
    'dx': True,
    'ent': True,
    'fld': False,
    'grd': False,
    'gro': True,
    'idx': True,
    'mae': False,
    'maegz': False,
    'map': False,
    'mmod': False,
    'mmtf': True,
    'moe': False,
    'mol': True,
    'mol2': True,
    'mrc': False,
    'mtz': True,
    'o': False,
    'omap': True,
    'p1m': False,
    'p5m': False,
    'pdb': True,
    'pdb1': True,
    'pdbqt': False,
    'ph4': False,
    'phi': False,
    'pkl': False,
    'pml': True,
    'pqr': True,
    'pse': True,
    'psw': True,
    'pwg': True,
    'py': False,
    'pym': True,
    'pze': True,
    'pzw': True,
    'r3d': True,
    'sdf': True,
    'spi': False,
    'top': False,
    'trj': False,
    'vdb': True,
    'xplor': True,
    'xyz': True,
}


def ext_get_primary(ext):
    r'''
    If `ext` is an alias for another extension, return the other, otherwise return `ext`
    '''
    try:
        from pymol import importing
    except ImportError:
        print('import pymol failed')
        return ext

    return importing.filename_to_format('.' + ext)[2]


def HKEY(allusers=False):
    return winreg.HKEY_LOCAL_MACHINE if allusers else winreg.HKEY_CURRENT_USER


def CLASSES(allusers=False, readmode=False):
    if readmode and not allusers:
        return winreg.HKEY_CLASSES_ROOT

    return winreg.CreateKey(HKEY(allusers), r"Software\Classes")


def get_icon(ico):
    r'''
    Get the fully qualified path to a file which may be found in Scripts,
    Menu, or $PYMOL_DATA.
    '''
    for filename in [
            os.path.join(sys.prefix, "Scripts", ico),
            os.path.join(sys.prefix, "Menu", ico),
            os.path.join(os.getenv("PYMOL_DATA", ""), "pymol", "icons", ico),
    ]:
        if os.path.exists(filename):
            return filename
    return ''


def get_exe():
    r'''
    Get the fully qualified path to the PyMOL executable
    '''
    for ext in ['.exe', '.bat']:
        exe = os.path.join(sys.prefix, "Scripts", "pymol" + ext)
        if os.path.exists(exe):
            return exe


def get_cmd():
    r'''
    Get the shell command to launch PyMOL with one argument
    '''
    exe = os.path.join(sys.prefix, "PyMOLWin.exe")
    if os.path.exists(exe):
        return r'"{}" "%1"'.format(exe)
    for exe in [os.path.join(sys.prefix, "pythonw.exe"), sys.executable]:
        if os.path.exists(exe):
            return r'"{}" -m pymol "%1"'.format(exe)
    exe = get_exe()
    if exe:
        return r'"{}" "%1"'.format(exe)
    raise OSError('pymol not found')


def deletesubkeys(key):
    try:
        while True:
            name = winreg.EnumKey(key, 0)
            deletekey(key, name, quiet=True)
    except OSError:
        pass


def deletekey(key, name, quiet=False):
    try:
        with winreg.OpenKey(key, name) as subkey:
            deletesubkeys(subkey)
    except OSError:
        return False
    else:
        if not quiet:
            print('Deleting ' + name)
        winreg.DeleteKey(key, name)
        return True


def is_value_equal(key, sub_key, subsub_check, value):
    try:
        v = winreg.QueryValue(key, sub_key + subsub_check)
    except OSError as e:
        return False

    if v != value:
        return False

    return True


def deletekey_if_match(key, sub_key, subsub_check, value):
    r'''
    Delete `sub_key` if `sub_key+subsub_check` (default) value is `value`
    '''
    if is_value_equal(key, sub_key, subsub_check, value):
        return deletekey(key, sub_key)

    return False


def cleanup(allusers=False):
    r'''
    Remove registry entries if their values match the expected value
    '''
    cmd = get_cmd()

    with CLASSES(allusers) as ckey:
        index = 0
        while True:
            try:
                progid = winreg.EnumKey(ckey, index)
            except OSError as e:
                break

            if not deletekey_if_match(ckey, progid, r"\shell\open\command",
                                      cmd):
                index += 1


def register_extensions(extensions=extensions, allusers=False):
    r'''
    Registers the given extensions as "PyMOL.<ext>" ProgIDs in the Windows
    registry.
    '''
    cmd = get_cmd()
    ico = get_icon('pymolfile.ico')
    ckey = CLASSES(allusers)
    cache = set()

    for ext in extensions:
        primary = ext_get_primary(ext)
        progid = 'PyMOL.' + primary
        is_text = ext in texttypes or primary in texttypes

        print('Registering Extension: ' + ext + ' -> ' + progid)

        deletekey(
            HKEY(allusers),
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\." +
            ext + r"\UserChoice")

        with winreg.CreateKey(ckey, '.' + ext) as key:
            winreg.SetValue(key, '', winreg.REG_SZ, progid)

        if progid in cache:
            continue

        cache.add(progid)

        with winreg.CreateKey(ckey, progid) as key:
            winreg.SetValue(key, r"shell\open\command", winreg.REG_SZ, cmd)
            winreg.SetValue(key, r"shell\open", winreg.REG_SZ,
                            'Open with PyMOL')

            if is_text:
                winreg.SetValue(key, r"shell\open", winreg.REG_SZ,
                                'Run with PyMOL')
                winreg.SetValueEx(key, r"PerceivedType", 0, winreg.REG_SZ,
                                  'text')

            if primary in ['pse']:
                winreg.SetValue(key, r"shell\play\command", winreg.REG_SZ,
                                cmd + ' -A6')
                winreg.SetValue(key, r"shell\play", winreg.REG_SZ,
                                'Present with PyMOL')

            if primary in ['psw']:
                editcmd = cmd[:-4] + ' -d "load %1"'
                winreg.SetValue(key, r"shell\edit\command", winreg.REG_SZ,
                                editcmd)
                winreg.SetValue(key, r"shell\edit", winreg.REG_SZ,
                                'Edit with PyMOL')

            if primary in descriptions:
                winreg.SetValueEx(key, r"FriendlyTypeName", 0, winreg.REG_SZ,
                                  descriptions[primary])

            if ico:
                winreg.SetValue(key, r"DefaultIcon", winreg.REG_SZ, ico)


def unregister_extension(ext, allusers=False):
    r'''
    Unregister the given extension (as good as possible)
    '''
    cmd = get_cmd()
    primary = ext_get_primary(ext)
    progid = 'PyMOL.' + primary

    if check_file_association(ext, cmd, allusers):
        print('Unregistering Extension: ' + ext)

        deletekey(
            HKEY(allusers),
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\." +
            ext + r"\UserChoice")

        deletekey(CLASSES(allusers), '.' + ext)

    try:
        with winreg.OpenKey(
                HKEY(allusers),
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\."
                + ext + r"\OpenWithProgIds", 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.DeleteValue(key, progid)
    except OSError as e:
        pass


def finalize():
    SHCNE_ASSOCCHANGED = 0x08000000
    try:
        ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, 0, None, None)
    except Exception as e:
        print(e)


def isAdmin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def check_file_association(ext, cmd, allusers=False):
    ckey = CLASSES(allusers, True)

    try:
        with winreg.OpenKey(
                HKEY(),
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\."
                + ext + r"\UserChoice") as uckey:
            progid = winreg.QueryValueEx(uckey, "Progid")[0]
    except OSError as e:
        try:
            progid = winreg.QueryValue(ckey, "." + ext)
        except OSError as e:
            return False

    return is_value_equal(ckey, progid, r"\shell\open\command", cmd)


################ GUI ############################################


def extensions_gui(parent=None):
    from pymol.Qt import QtCore, QtGui, QtWidgets
    from pymol.Qt.utils import PopupOnException
    Qt = QtCore.Qt

    def connectItemChangedDec(func):

        def wrapper(self, *args, **kwargs):
            try:
                self.model.itemChanged.disconnect()
            except Exception as e:
                pass

            try:
                return func(self, *args, **kwargs)
            finally:
                self.model.itemChanged.connect(self.itemChanged)

        return wrapper

    class PyMOLExtensionsHelper(QtWidgets.QWidget):

        def __init__(self):
            QtWidgets.QWidget.__init__(self, parent, Qt.Window)
            self.setMinimumSize(400, 500)
            self.setWindowTitle('Register File Extensions')

            self.model = QtGui.QStandardItemModel(self)

            layout = QtWidgets.QVBoxLayout(self)
            self.setLayout(layout)

            label = QtWidgets.QLabel(
                "Select file types to register them with PyMOL", self)
            layout.addWidget(label)

            alluserslayout = QtWidgets.QHBoxLayout()
            alluserslayout.setObjectName("alluserslayout")
            layout.addLayout(alluserslayout)

            buttonlayout = QtWidgets.QHBoxLayout()
            buttonlayout.setObjectName("buttonlayout")
            layout.addLayout(buttonlayout)

            self.table = QtWidgets.QTableView(self)
            self.table.setModel(self.model)
            layout.addWidget(self.table)

            button = QtWidgets.QPushButton("Register Recommended (*)", self)
            buttonlayout.addWidget(button)
            button.pressed.connect(self.setRecommended)

            button = QtWidgets.QPushButton("Register All", self)
            buttonlayout.addWidget(button)
            button.pressed.connect(self.setAll)

            button = QtWidgets.QPushButton("Clear", self)
            button.setToolTip("Clean up Registry")
            buttonlayout.addWidget(button)
            button.pressed.connect(self.clear)

            if isAdmin():
                r0 = QtWidgets.QRadioButton("Only for me")
                r0.setToolTip("HKEY_CURRENT_USER registry branch")
                r0.setChecked(True)
                r1 = QtWidgets.QRadioButton("For all users")
                r1.setToolTip("HKEY_LOCAL_MACHINE registry branch")
                allusersgroup = QtWidgets.QButtonGroup(self)
                allusersgroup.addButton(r0)
                allusersgroup.addButton(r1)
                allusersgroup.buttonClicked.connect(self.populateData)
                alluserslayout.addWidget(r0)
                alluserslayout.addWidget(r1)
                alluserslayout.addStretch()
                self.allusersbutton = r1
            else:
                self.allusersbutton = None

            self.finalize_timer = QtCore.QTimer()
            self.finalize_timer.setSingleShot(True)
            self.finalize_timer.setInterval(500)
            self.finalize_timer.timeout.connect(finalize)

            self.populateData()

            # keep reference to window, otherwise Qt will auto-close it
            self._self_ref = self

        def closeEvent(self, event):
            self._self_ref = None
            super(PyMOLExtensionsHelper, self).closeEvent(event)

        def allusers(self):
            if self.allusersbutton is None:
                return False
            return self.allusersbutton.isChecked()

        @connectItemChangedDec
        def populateData(self, _button=None):
            """
            Fill the model with data from PyMOL
            """
            QSI = QtGui.QStandardItem  # For brevity
            cmd = get_cmd()

            self.model.clear()

            for ext in sorted(extensions):
                name_item = QSI(ext.upper())
                name_item.setFlags(Qt.ItemIsEnabled)

                desc_item = QSI(
                    descriptions.get(
                        ext, descriptions.get(ext_get_primary(ext), '')))

                desc_item.setFlags(Qt.ItemIsEnabled)

                recom_item = QSI('*' if extensions[ext] else '')
                recom_item.setFlags(Qt.ItemIsEnabled)

                value_item = QSI()
                value_item.setData(ext)
                value_item.setCheckable(True)
                value_item.setEditable(False)  # Can't edit text (but toggles)
                if check_file_association(ext, cmd, self.allusers()):
                    value_item.setCheckState(Qt.Checked)

                self.model.appendRow(
                    [value_item, name_item, desc_item, recom_item])

            self.formatTable()

        def setAll(self):
            for i in range(len(extensions)):
                item = self.model.item(i)
                item.setCheckState(Qt.Checked)

            self.finalize_timer.start()

        def setRecommended(self):
            for i in range(len(extensions)):
                item = self.model.item(i)
                ext = self.itemGetData(item)
                if extensions[ext]:
                    item.setCheckState(Qt.Checked)

            self.finalize_timer.start()

        @PopupOnException.decorator
        @connectItemChangedDec
        def clear(self):
            for i in range(len(extensions)):
                item = self.model.item(i)
                item.setCheckState(Qt.Unchecked)

            cleanup(self.allusers())
            self.finalize_timer.start()

        def formatTable(self):
            """
            Set up the table to look appropriately
            """
            hh = self.table.horizontalHeader()
            hh.setVisible(False)
            hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)
            self.table.resizeColumnsToContents()

        def itemGetData(self, item):
            try:
                return item.data().toString()
            except AttributeError:  # PySide not PyQT, different QVariant handling
                return item.data()

        @PopupOnException.decorator
        def itemChanged(self, item):
            ext = self.itemGetData(item)

            if item.checkState() == Qt.Checked:
                register_extensions([ext], self.allusers())
            else:
                unregister_extension(ext, self.allusers())

            self.finalize_timer.start()

    if QtWidgets.QApplication.instance() is None:
        app = QtWidgets.QApplication([])
    else:
        app = None

    w = PyMOLExtensionsHelper()
    w.show()

    if app is None:
        return w

    app.exec_()


def prompt_user_once(parent=None):
    '''
    Show the filetypes dialog on the first run
    '''
    if is_first_run('winfiletypes'):
        extensions_gui(parent)


def is_first_run(checkname, identifier=sys.prefix):
    '''
    Return False if installation `identifier` is found in
    "~/.pymol/`checkname`_done.txt", otherwise update the
    file and return True.
    '''
    checkfile = os.path.join(
        os.path.expanduser('~'), '.pymol', checkname + '_done.txt')

    try:
        with open(checkfile, 'r') as handle:
            prefixes = handle.readlines()
    except EnvironmentError:
        prefixes = []
    except Exception as e:
        # UnicodeDecodeError if previously saved with a different encoding
        print(e)
        prefixes = []

    prefix = identifier + '\n'

    if prefix in prefixes:
        return False

    # add new prefix and remove obsolete ones
    prefixes = [prefix] + [p for p in prefixes if os.path.isdir(p.rstrip())]

    try:
        os.makedirs(os.path.dirname(checkfile))
    except EnvironmentError:
        pass

    try:
        with open(checkfile, 'w') as handle:
            handle.writelines(prefixes)
    except EnvironmentError:
        return False

    # confirmed that we have write access to the checkfile and that prefix for
    # this installation was not in there yet
    return True


###################### Register with Maestro ####################


def register_with_maestro(allusers=False):
    key_name = r"Software\Schrodinger\PyMOL\PYMOL_PATH"
    try:
        with winreg.CreateKey(HKEY(allusers), key_name) as key:
            winreg.SetValue(key, '', winreg.REG_SZ, sys.prefix)
    except OSError as e:
        print('Registry update failed:', e)
    else:
        print('Updated registry:', key_name)


def register_with_maestro_gui(parent=None):
    from pymol.Qt import QtWidgets
    QMB = QtWidgets.QMessageBox
    ok = QMB.question(parent, "Register with Maestro?",
            'Register this PyMOL installation for '
            'the "Send to PyMOL" feature in Maestro?',
            QMB.Ok | QMB.Cancel, QMB.Ok)
    if ok == QMB.Ok:
        register_with_maestro()


###################### CLI ######################################


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--allusers', action="store_true", help="Register for all users")
    parser.add_argument(
        '--all', action="store_true", help="Register all extensions")
    parser.add_argument(
        '--recommended',
        action="store_true",
        help="Register recommended extensions")
    parser.add_argument(
        '--clear', action="store_true", help="Clean up registry")
    parser.add_argument(
        'exts', nargs="*", help="Extensions (e.g. pdb cif mmtf)")
    options = parser.parse_args()

    if options.allusers and not isAdmin():
        # Registry write access for "all users" (HKEY_LOCAL_MACHINE) is
        # only available to admins or to "elevated" users (Windows prompts
        # the user for permission)
        quote = lambda s: u'"' + s + u'"'
        try:
            r = ctypes.windll.shell32.ShellExecuteW(
                None, u'runas', u'' + sys.executable,
                u' '.join(map(quote, sys.argv)), None, 1)
            if r > 32:
                # ShellExecute returns a value greater than 32 on success.
                # We've spawned execution of this script to the elevated
                # process, so exit this one.
                sys.exit()
            print('Elevation failed with status', r)
        except Exception as e:
            print(e)

    if options.all:
        options.exts.extend(extensions)
    elif options.recommended:
        options.exts.extend(e for (e, r) in extensions.items() if r)
    elif options.clear:
        cleanup(options.allusers)
        finalize()
    elif not options.exts:
        extensions_gui()

    if options.exts:
        register_extensions(options.exts, options.allusers)
        finalize()


if __name__ == '__main__':
    main()
