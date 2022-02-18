import sys
import os
import pymol
cmd = sys.modules["pymol.cmd"]
from pymol.undo import undoSessionDecorator


def _get_stereo_labels_rdkit(molstr, propgetter):
    from rdkit import Chem

    mol = Chem.MolFromMolBlock(molstr, False, False)

    # if kekulizing fails, then e.g. [R1]P(=O)([O-])[R2] will be chiral.
    flags_warn = (
            Chem.SanitizeFlags.SANITIZE_KEKULIZE |
            Chem.SanitizeFlags.SANITIZE_PROPERTIES)

    # raises an exception on invalid input
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL & ~flags_warn)

    # warns about invalid input, but keeps going
    try:
        Chem.SanitizeMol(mol, flags_warn)
    except ValueError as e:
        print(' RDKit-Warning: ' + str(e).strip())

    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, True, True)

    return [propgetter(a) for a in mol.GetAtoms()]


def get_stereo_labels_rdkit_cip(molstr):
    '''R/S labels with RDKit backend
    '''
    func = lambda a: a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else ''
    return _get_stereo_labels_rdkit(molstr, func)


def get_stereo_labels_rdkit_parity(molstr):
    '''parity labels with RDKit backend
    '''
    from rdkit import Chem
    mapping = {
        Chem.CHI_TETRAHEDRAL_CW: 'odd',
        Chem.CHI_TETRAHEDRAL_CCW: 'even',
        Chem.CHI_OTHER: '?',
    }
    func = lambda a: mapping.get(a.GetChiralTag(), '')
    return _get_stereo_labels_rdkit(molstr, func)


def get_stereo_labels_schrodinger(molstr):
    '''R/S labels with SCHRODINGER backend
    '''
    if 'SCHRODINGER' not in os.environ:
        raise pymol.CmdException('SCHRODINGER environment variable not set')

    import subprocess
    import tempfile

    schrun = os.path.join(os.environ['SCHRODINGER'], 'run')
    script = os.path.join(os.path.dirname(__file__), 'schrodinger-helper.py')
    filename = tempfile.mktemp('.mol')
    args = [schrun, script, filename]

    env = dict(os.environ)
    env.pop('PYTHONEXECUTABLE', '')  # messes up on Mac

    if pymol.IS_WINDOWS:
        # Fix for 2020-4 (PYMOL-3572)
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)

    with open(filename, 'w') as handle:
        handle.write(molstr)

    try:
        p = subprocess.Popen(args, env=env,
                universal_newlines=True,
                stdin=subprocess.PIPE, # Windows fix
                stderr=subprocess.PIPE, # Windows fix
                stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
    finally:
        os.unlink(filename)

    if stderr:
        print(stderr)

    return stdout.splitlines()


methods = {
    'rdkit': get_stereo_labels_rdkit_cip,
    'rdkitparity': get_stereo_labels_rdkit_parity,
    'schrodinger': get_stereo_labels_schrodinger,
}


@undoSessionDecorator("Assign Stereo")
def assign_stereo(selection='all', state=-1, method='', quiet=1, prop='stereo',
        _self=cmd):
    '''
DESCRIPTION

    Assign "stereo" atom property (R/S stereochemistry).

    Requires either a Schrodinger Suite installation (SCHRODINGER
    environment variable set) or RDKit (rdkit Python module).

USAGE

    assign_stereo [selection [, state [, method ]]]

ARGUMENTS

    selection = str: atom selection {default: all}

    state = int: object state {default: -1 (current)}

    method = schrodinger or rdkit: {default: try both}
    '''
    state, quiet = int(state), int(quiet)

    # method check
    if method and method not in methods:
        raise pymol.CmdException("method '{}' not in {}".format(method, list(methods)))

    # alt code check
    alt_codes = set()
    cmd.iterate('({})&!alt ""'.format(selection),
            'alt_codes.add(alt)', space={'alt_codes': alt_codes})
    if len(alt_codes) > 1:
        alt_codes = sorted(alt_codes)
        selection = '({}) & alt +{}'.format(selection, alt_codes[0])
        print(' Warning: only using first alt code of: {}'.format(alt_codes))

    molstr = _self.get_str('mol', selection, state)

    if method:
        labels = methods[method](molstr)
    else:
        for method in ('rdkit', 'schrodinger'):
            try:
                labels = methods[method](molstr)
            except BaseException as e:
                print(' Method "{}" not available ({})'.format(method, str(e).strip()))
            else:
                print(' Using method={}'.format(method))
                break
        else:
            raise pymol.CmdException('need SCHRODINGER or rdkit')

    # apply new labels
    _self.alter_state(state, selection, prop + ' = next(label_iter)',
            space={'label_iter': iter(labels), 'next': next})
