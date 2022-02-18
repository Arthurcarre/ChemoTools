import functools
import sys

import pymol
cmd = sys.modules["pymol.cmd"]
from pymol import _cmd

class UndoMode:
    Disable = 0
    Enable = 1
    Unpause = 2
    Pause = 3
    PermanentDisable = 4

class UndoGroupMode:
    Close = 0
    Open = 1

class UndoGroupCM(object):
    def __init__(self, group_name="", _self=cmd):
        self.cmd = _self
        self.group_name = group_name
    def __enter__(self):
        _cmd.undo_group_open(self.cmd._COb, self.group_name)
    def __exit__(self, exc_type, exc_value, traceback):
        _cmd.undo_group_close(self.cmd._COb)

def undoGroupDecorator(group_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, _self=cmd, **kwargs):
            with _self.UndoGroupCM(group_name):
                return func(*args, **kwargs, _self=_self)
        return wrapper
    return decorator

def undo_copy_sele(sele, mode, originalMode, cmd_name, _self=cmd):
    '''
    Makes a copy of the selected object or prepares a restoration depending on mode

    :param sele: selection

    :param mode (0 or 1): (1 = forward) makes copy of selection;
                   (0 = reverse) restores copy of selection

    :param originalMode: original Undo Manager mode

    :param cmd_name: Name of the undo group. This is typically the name of the undoable command.
    '''
    with _self.lockcm:
        return _cmd.undo_copy_sele(_self._COb, sele, int(mode), int(originalMode), cmd_name)

class UndoCopySeleCM:
    '''
    Context manager that copies a given selection upon entry and exit
    '''
    def __init__(self, sele, cmd_name, _self=cmd):
        self.cmd = _self
        self.cmd_name = cmd_name
        self.sele = sele
        self.fwd = 1
        self.rev = 0
        self.originalMode = _self.undo_get_mode()
    def __enter__(self):
        self.cmd.undo_mode(UndoMode.Unpause)
        self.cmd.undo_copy_sele(self.sele, self.fwd, self.originalMode, self.cmd_name)
        self.cmd.undo_mode(UndoMode.Pause)
    def __exit__(self, exc_type, exc_value, traceback):
        self.cmd.undo_mode(UndoMode.Unpause)
        self.cmd.undo_copy_sele(self.sele, self.rev, self.originalMode, self.cmd_name)
        self.cmd.undo_restore_mode(self.originalMode)

def undo_copy_obj(sele, mode, original_mode, cmd_name, _self=cmd):
    '''
    Makes a copy of the selected object or prepares a restoration depending on mode.

    :param sele: selection

    :param mode (0 or 1): (1 = forward) makes copy of selection;
                   (0 = reverse) restores copy of selection

    :param original_mode (0 or 1): undo mode which this function was first called

    :param cmd_name: Name of the undo group. This is typically the name of the undoable command.
    '''
    with _self.lockcm:
        return _cmd.undo_copy_obj(_self._COb, sele, int(mode), int(original_mode), cmd_name)

class UndoCopyObjCM:
    '''
    Context manager that copies a given selection upon entry and exit
    '''
    def __init__(self, sele, cmd_name, _self=cmd):
        self.cmd = _self
        self.cmd_name = cmd_name
        self.sele = sele
        self.fwd = 1
        self.rev = 0
        self.originalMode = _self.undo_get_mode()
        obj_list = self.cmd.get_object_list(sele)
        pre_str = f"?{obj_list[0]}" if obj_list else ""
        self.obj_name = f"{pre_str} {sele}"
    def __enter__(self):
        self.cmd.undo_mode(UndoMode.Unpause)
        self.cmd.undo_copy_obj(self.sele, self.fwd, self.originalMode, self.cmd_name)
        self.cmd.undo_mode(UndoMode.Pause)
    def __exit__(self, exc_type, exc_value, traceback):
        self.cmd.undo_mode(UndoMode.Unpause)
        self.cmd.undo_copy_obj(self.obj_name, self.rev, self.originalMode, self.cmd_name)
        self.cmd.undo_restore_mode(self.originalMode)

class UndoPauseCM:
    '''
    Pauses the Undo Manager temporarily and restores the original mode
    upon exit
    '''
    def __init__(self, _self=cmd):
        self.cmd = _self
        self.originalMode = _self.undo_get_mode()
    def __enter__(self):
        self.cmd.undo_mode(UndoMode.Pause)
    def __exit__(self, exc_type, exc_value, traceback):
        self.cmd.undo_mode(self.originalMode)

def undoPauseDecorator():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, _self=cmd, **kwargs):
            with _self.UndoPauseCM():
                return func(*args, **kwargs, _self=_self)
        return wrapper
    return decorator

def undo_session_cm(cmd_name, mode, originalMode, _self=cmd):
    '''
    Creates an undo session for context manager
    '''
    with _self.lockcm:
        return _cmd.undo_session_cm(_self._COb, mode, originalMode, cmd_name)

class UndoSessionCM(object):
    FWD = 1
    REV = 0
    def __init__(self, group_name="", _self=cmd):
        self.cmd = _self
        self.group_name = group_name
        self.originalMode = _self.undo_get_mode()
    def __enter__(self):
        self.cmd.undo_session_cm(self.group_name, self.FWD, self.originalMode)
    def __exit__(self, exc_type, exc_value, traceback):
        self.cmd.undo_session_cm(self.group_name, self.REV, self.originalMode)
        self.cmd.undo_restore_mode(self.originalMode)

def undoSessionDecorator(group_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, _self=cmd, **kwargs):
            with _self.UndoSessionCM(group_name):
                return func(*args, **kwargs, _self=_self)
        return wrapper
    return decorator

def undo_mode(mode, _self=cmd):
    '''
    Sets the Undo Manager activation mode

    :param mode: enable, disable, or pause (See class UndoMode)
    '''
    with _self.lockcm:
        return _cmd.undo_mode(_self._COb, mode)

def undo_get_mode(_self=cmd):
    '''
    Retrieves the Undo Manager activation mode
    '''
    with _self.lockcm:
        return _cmd.undo_get_mode(_self._COb)

def undo_status(_self=cmd):
    '''
    Prints the current status of the Undo Manager
    '''
    with _self.lockcm:
        return _cmd.undo_status(_self._COb)

def undo_current_undo_redo(_self=cmd):
    '''
    Prints the next undoable command name
    '''
    with _self.lockcm:
        return _cmd.undo_current_undo_redo(_self._COb)

def undo_clear_cache(_self=cmd):
    '''
    Clears the Undo Manager
    '''
    with _self.lockcm:
        return _cmd.undo_clear_cache(_self._COb)

def undo_is_enabled(_self=cmd):
    '''
    Queries whether the Undo Manager is enabled
    '''
    return not _self.undo_is_disabled()

def undo_is_disabled(_self=cmd):
    '''
    Queries whether the Undo Manager is disabled
    '''
    mode = _self.undo_get_mode()
    return mode in (UndoMode.Disable, UndoMode.PermanentDisable)

def undo_enable(_self=cmd):
    '''
    Enables the Undo Manager
    '''
    _self.undo_mode(UndoMode.Enable)

def undo_disable(_self=cmd):
    '''
    Disables the Undo Manager
    '''
    _self.undo_mode(UndoMode.Disable)

def undo_restore_mode(original_mode, _self=cmd):
    '''
    Restores the Undo Manager mode if not disabled.
    :param originalmode: mode to restore
    '''
    if _self.undo_get_mode() != UndoMode.Disable:
        _self.undo_mode(original_mode)