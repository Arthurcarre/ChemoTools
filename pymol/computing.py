#A* -------------------------------------------------------------------
#B* This file contains source code for the PyMOL computer program
#C* Copyright (c) Schrodinger, LLC
#D* -------------------------------------------------------------------
#E* It is unlawful to modify or remove this copyright notice.
#F* -------------------------------------------------------------------
#G* Please see the accompanying LICENSE file for further information.
#H* -------------------------------------------------------------------
#I* Additional authors of this source file include:
#-*
#-*
#-*
#Z* -------------------------------------------------------------------

import sys
cmd_module = __import__("sys").modules["pymol.cmd"]
from .cmd import _cmd, lock, unlock, Shortcut, \
     _feedback, fb_module, fb_mask, \
     DEFAULT_ERROR, DEFAULT_SUCCESS, _raising, is_ok, is_error, \
     is_list, safe_list_eval, is_string

import traceback
import threading
import os
import pymol
from pymol.undo import UndoMode

def model_to_sdf_list(self_cmd,model):
    from chempy import io

    sdf_list = io.mol.toList(model)
    fixed = []
    restrained = []
    at_id = 1
    for atom in model.atom:
        if atom.flags & 4:
            if hasattr(atom,'ref_coord'):
                restrained.append( [at_id,atom.ref_coord])
        if atom.flags & 8:
            fixed.append(at_id)
        at_id = at_id + 1
    fit_flag = 1
    if len(fixed):
        fit_flag = 0
        sdf_list.append(">  <FIXED_ATOMS>\n")
        sdf_list.append("+ ATOM\n");
        for ID in fixed:
            sdf_list.append("| %4d\n"%ID)
        sdf_list.append("\n")
    if len(restrained):
        fit_flag = 0
        sdf_list.append(">  <RESTRAINED_ATOMS>\n")
        sdf_list.append("+ ATOM    MIN    MAX F_CONST         X         Y         Z\n")
        for entry in restrained:
            xrd = entry[1]
            sdf_list.append("| %4d %6.3f %6.3f %6.3f %10.4f %10.4f %10.4f\n"%
                            (entry[0],0,0,3,xrd[0],xrd[1],xrd[2]))
        sdf_list.append("\n")
    electro_mode = int(self_cmd.get('clean_electro_mode'))
    if electro_mode == 0:
        fit_flag = 0
        sdf_list.append(">  <ELECTROSTATICS>\n")
        sdf_list.append("+ TREATMENT\n")
        sdf_list.append("| NONE\n")
        sdf_list.append("\n")
    sdf_list.append("$$$$\n")
    return (fit_flag, sdf_list)

def get_energy_from_rec(rec):
    # we really need to replace this with a proper SD parser...
    result = 9999.00
    try:
        rec_list = rec.splitlines()
        read_energy = 0
        for line in rec_list:
            if read_energy == 1:
                result = float(line.strip())
                break
            if line.strip() == '> <MMFF94 energy>':
                read_energy = 1
    except:
        traceback.print_exc()
    return result


class mengine:
    '''
    Replacement for freemol.mengine
    '''
    _mengine_exe = None

    @classmethod
    def validate(cls):
        '''
        @return True if mengine executable was found
        '''
        if cls._mengine_exe is None:
            from shutil import which
            cls._mengine_exe = which('mengine') or which('mengine.exe')
        return cls._mengine_exe is not None

    @classmethod
    def run(cls, sdfstr):
        '''
        @param sdfstr Input molecule as SDF string
        @return (stdout, stderr) tuple or None if mengine did not run
        '''
        assert cls._mengine_exe is not None

        if not isinstance(sdfstr, bytes):
            sdfstr = sdfstr.encode('utf-8')

        import subprocess
        kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if pymol.IS_WINDOWS:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

        try:
            p = subprocess.Popen([cls._mengine_exe], **kwargs)
            return p.communicate(sdfstr)
        except:
            print("Error: mengine did not run")
            traceback.print_exc()


class CleanJob:
    def __init__(self,self_cmd,sele,state=-1,message=None):
        self.cmd = self_cmd
        if message == '':
            message = None
        if state<1:
            state = self_cmd.get_state()
        # this code will moved elsewhere
        self.ok = 1

        if not mengine.validate():
                self.ok = 0
                print("Error: Unable to validate freemol.mengine")
                return

        if self_cmd.count_atoms(sele) > 999:
                self.ok = 0
                print("Error: Sorry, clean is currently limited to 999 atoms")
                return

        if True:
            if message is not None:
                self.cmd.do("_ cmd.wizard('message','''%s''')"%message)
            obj_list = self_cmd.get_object_list("bymol ("+sele+")")
            self.ok = 0
            result = None
            input_model = None
            if is_list(obj_list) and (len(obj_list)==1):
                obj_name = obj_list[0]
                self_cmd.sculpt_deactivate(obj_name)
                # eliminate all sculpting information for object
                self.cmd.sculpt_purge()
                self.cmd.set("sculpting",0)
                if self_cmd.count_atoms(obj_name+" and flag 2"): # any atoms restrained?
                    self_cmd.reference("validate",obj_name,state) # then we have reference coordinates
                input_model = self_cmd.get_model(obj_name,state=state)
                (fit_flag, sdf_list) = model_to_sdf_list(self_cmd,input_model)
                input_sdf = ''.join(sdf_list)
                input_sdf = input_sdf.encode()

                result = mengine.run(input_sdf)
                if result is not None:
                    if len(result):
                        clean_sdf = result[0]
                        clean_sdf = clean_sdf.decode()

                        clean_rec = clean_sdf.split("$$$$")[0]
                        clean_name = ""
                        self.energy = get_energy_from_rec(clean_rec)
                        try:
                            if len(clean_rec) and int(self.energy) != 9999:
                                clean_name = self_cmd.get_unused_name("builder_clean_tmp")
                                self_cmd.set("suspend_updates")
                                self.ok = 1
                            else:
                                self.ok = 0

                            if self.ok:
                                self_cmd.read_molstr(clean_rec, clean_name, zoom=0)
                                # need to insert some error checking here
                                if clean_name in self_cmd.get_names("objects"):
                                    self_cmd.set("retain_order","1",clean_name)
                                    if fit_flag:
                                        self_cmd.fit(clean_name, obj_name, matchmaker=4,
                                                     mobile_state=1, target_state=state)
                                    self_cmd.update(obj_name, clean_name, matchmaker=0,
                                                    source_state=1, target_state=state)
                                    self_cmd.sculpt_activate(obj_name)
                                    self_cmd.sculpt_deactivate(obj_name)
                                    self.ok = 1
                                    message = "Clean: Finished. Energy = %3.2f" % self.energy
                                    if message is not None:
                                        self.cmd.do("_ cmd.wizard('message','''%s''')"%message)
                                        self.cmd.do("_ wizard")
                        except ValueError:
                            self.ok = 0
                        finally:
                            self_cmd.delete(clean_name)
                            self_cmd.unset("suspend_updates")
            if not self.ok:
                # we can't call warn because this is the not the tcl-tk gui thread
                if result is not None:
                    if len(result)>1:
                        errormsg = result[1]
                        errormsg = errormsg.decode(errors='replace')
                        print("\n=== mengine errors below === ")
                        print(errormsg.strip())
                        print("=== mengine errors above ===\n")
                failed_file = "cleanup_failed.sdf"
                print("Clean-Error: Structure cleanup failed.  Invalid input or software malfuction?")
                aromatic = 0
                if input_model is not None:
                    aromatic = any(bond.order == 4 for bond in input_model.bond)
                try:
                    open(failed_file,'wb').write(input_sdf)
                    print("Clean-Error: Wrote SD file '%s' into the directory:"%failed_file)
                    print("Clean-Error: '%s'."%os.getcwd())
                    print("Clean-Error: If you believe PyMOL should be able to handle this structure")
                    print("Clean-Error: then please email that SD file to help@schrodinger.com. Thank you!")
                except IOError:
                    print("Unabled to write '%s"%failed_file)
                if aromatic:
                    print("Clean-Warning: Please eliminate aromatic bonds and then try again.")
        if message is not None:
            self_cmd.do("_ wizard")

def _clean(selection, present='', state=-1, fix='', restrain='',
          method='mmff', save_undo=1, message=None,
          _self=cmd_module):
    import uuid
    from . import colorprinting

    self_cmd = _self

    suffix = str(uuid.uuid4())
    clean1_sele = "_clean1_tmp" + suffix
    clean2_sele = "_clean2_tmp" + suffix
    clean_obj = "_clean_obj" + suffix
    r = DEFAULT_SUCCESS
    c = None
    with self_cmd.UndoPauseCM():
        if self_cmd.select(clean1_sele,selection,enable=0,state=state) < 1:
            colorprinting.warning("selection has no atoms in state {}".format(
                self_cmd.get_state() if state == -1 else state))
            self_cmd.undo_mode(UndoMode.Unpause)
            return None;

    with self_cmd.UndoCopySeleCM(selection, "Clean"):
        try:
            if present=='':
                self_cmd.select(clean2_sele," byres ("+clean1_sele+" extend 1)",enable=0,state=state) # go out 1 residue
            else:
                self_cmd.select(clean2_sele, clean1_sele+" or ("+present+")",enable=0,state=state)

            self_cmd.set("suspend_updates")
            self_cmd.rename(clean2_sele) # ensure identifiers are unique
            self_cmd.create(clean_obj, clean2_sele, zoom=0, source_state=state,target_state=1)
            self_cmd.disable(clean_obj)
            self_cmd.unset("suspend_updates")

            self_cmd.flag(3,clean_obj+" in ("+clean2_sele+" and not "+clean1_sele+")","set")
            # fix nearby atoms

            # sort hydrogens to the end (PYMOL-2985)
            self_cmd.alter(clean_obj, 'rank = (protons == 1)')

            self_cmd.h_add(clean_obj) # fill any open valences

            if message is None:
                at_cnt = self_cmd.count_atoms(clean_obj)
                message = 'Clean: Cleaning %d atoms.  Please wait...'%at_cnt

            c = CleanJob(self_cmd, clean_obj, 1, message=message)

            if c.ok:
                self_cmd.update(clean1_sele, clean_obj,
                                source_state=1, target_state=state)
        finally:
            self_cmd.delete(clean_obj)
            self_cmd.delete(clean1_sele)
            self_cmd.delete(clean2_sele)

    if hasattr(c,"energy"):
        return c.energy
    else:
        return None

def clean(selection, present='', state=-1, fix='', restrain='',
          method='mmff', async_=0, save_undo=1, message=None,
          _self=cmd_module, **kwargs):
    '''
DESCRIPTION

    Note: This operation is limited to 999 atoms.

    Run energy minimization on the given selection, using an MMFF94
    force field.

ARGUMENTS

    selection = str: atom selection to minimize

    present = str: selection of fixed atoms to restrain the minimization

    state = int: object state {default: -1 (current)}

    fix = UNUSED

    restraing = UNUSED

    method = UNUSED

    async = 0/1: run in separate thread {default: 0}

    save_undo = UNUSED

    message = Message to display during async minimization

EXAMPLE

    # minimize ligand in binding pocket
    clean organic, all within 8 of organic
    '''
    state = int(state)
    if state == 0:
        raise pymol.CmdException('cleaning all states not supported')

    async_ = int(kwargs.pop('async', async_))

    if kwargs:
        raise pymol.CmdException('unknown argument: ' + ', '.join(kwargs))

    args = (selection, present, state, fix, restrain, method, save_undo, message, _self)

    if not async_:
        return _clean(*args)
    else:
        try:
            t = threading.Thread(target=_clean,
                             args=args)
            t.setDaemon(1)
            t.start()
        except:
            traceback.print_exc()
        return 0
