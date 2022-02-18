##
## Incentive PyMOL licensing helpers
##

import sys
import os
import threading

USE_DIRECTORIES = False
USE_FILENAMES = True

LICENCE_DIRECTORY = u'$PYMOL_PATH/licenses'
LICENCE_DIRECTORY_USER = u'~/.pymol/licenses'
LICENCE_FILENAME = u'$PYMOL_PATH/license.lic'
LICENCE_FILENAME_USER = u'~/.pymol/license.lic'

post_ns = {}


from os.path import expanduser


def post_license_check(msg):
    try:
        post_ns['callback'](msg)
    except KeyError:
        post_ns['message'] = msg


def set_post_license_check_handler(callback):
    post_ns['callback'] = callback
    try:
        callback(post_ns.pop('message'))
    except KeyError:
        pass


def get_license_search_path():
    '''Get the search path for license files'''
    path = []

    if USE_DIRECTORIES:
        path += [
            expanduser(LICENCE_DIRECTORY_USER),
            os.path.expandvars(LICENCE_DIRECTORY),
        ]

    if USE_FILENAMES:
        path += [filename for filename in [
            get_license_filename_user(),
            os.path.expandvars(LICENCE_FILENAME),
        ] if os.path.exists(filename)]

    # fallback on macOS: Schrodinger licenses directory (putting a license
    # file inside the app bundle is a Gatekeeper violation)
    if sys.platform == 'darwin':
        path.append(u'/Library/Application Support/Schrodinger/licenses')

    return os.pathsep.join(path)


def get_license_filenames():
    '''List of all license files which might get checked. Only for
    diagnostics feedback.
    '''
    import glob
    filenames = []
    for n in get_license_search_path().split(os.pathsep):
        if os.path.isfile(n):
            filenames.append(n)
        else:
            filenames.extend(glob.glob(n + os.sep + '*.lic'))
    return filenames


def get_license_filename_user():
    '''The user's license file location (existing or not)'''
    filename = os.path.expandvars(LICENCE_FILENAME_USER)
    filename = expanduser(filename)
    return filename


def install_license_key(key):
    '''Copy the given key to the user's license file and check the license'''
    licfile = get_license_filename_user()

    try:
        os.makedirs(os.path.dirname(licfile), 0o750)
    except OSError:
        pass

    try:
        os.unlink(licfile)
    except OSError:
        pass

    try:
        handle = open(licfile, 'wb')
    except IOError:
        print(' Warning: cannot write to "%s", using temporary file instead.' %
              licfile)
        import tempfile
        licfile = tempfile.mktemp('.lic')
        handle = open(licfile, 'wb')

    if not isinstance(key, bytes):
        key = key.encode("utf-8")

    # Unicode/UTF-8 BOM
    if key.startswith(b"\xEF\xBB\xBF"):
        key = key[3:]

    with handle:
        handle.write(key)

    return check_license_file(licfile)


def install_license_file(filename):
    '''Copy the given file to the user's home and check the license'''
    return install_license_key(open(filename, 'rb').read())


def get_site_license_key():
    '''Download and return the license key of an IP-based site license, or
    an empty string if IP not recognized or an error occured.'''
    try:
        import requests
        response = requests.get('https://pymol.org/getlicfile.php')
        response.raise_for_status()
    except Exception as e:
        print('internet request failed: ' + str(e))
        return ''
    return response.json().get('key', '')


def print_info(info=None, withnotice=True):
    '''Print various license info.

    @param withnotice: Show the notice, which may contain the invoice number
    '''
    if info is None:
        info = get_info()

    if 'date' in info:
        print(' License Expiry date: %s' % (info['date']))

    if withnotice and 'notice' in info:
        print(' License Notice: %s' % (info['notice']))

    if 'days' in info:
        print(' Warning: License will expire in %s days' % (info['days']))

    if 'msg' in info:
        print(info['msg'])


def get_info():
    '''Get a dictionary with license metadata'''
    import pymol
    return pymol._cmd.check_license_file('')


def check_license_file(licfile=''):
    '''Return an empty string on successful license check, or an error message
    on failure.'''
    if not licfile:
        licfile = get_license_search_path()

    # Workaround for FlexLM limitation (PYMOL-3507)
    if '@' in licfile and os.path.exists(licfile):
        with open(licfile) as handle:
            licfile = ("START_LICENSE\n" + handle.read().rstrip() +
                       "\nEND_LICENSE\n")

    import pymol
    msg = pymol._cmd.check_license_file(licfile)
    post_license_check(msg)

    # for rigimol
    os.environ['PYMOL_LICENSE_FILE'] = licfile


# check license in background
_t = threading.Thread(target=check_license_file)
_t.setDaemon(1)
_t.start()