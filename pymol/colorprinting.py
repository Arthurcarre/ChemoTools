##
## (c) Schrodinger, Inc.
##

import re

_RE_ANSICOLOR = re.compile('\033' r'\[(\d*)m')

ANSI_SUPPORT = False


def _ansicolors2html_repl(m):
    d = m.group(1) or '0'
    return {
        '0': '</span>',
        '31': '<span style="color:#f66">',  # red
        '32': '<span style="color:#3c3">',  # green
        '33': '<span style="color:#ff0">',  # yellow
        '34': '<span style="color:#99f">',  # blue
        '35': '<span style="color:#f6f">',  # magenta
        '36': '<span style="color:#6ff">',  # cyan
    }.get(d, '<span>UNKNOWN ' + d)


def ansicolors2html(s):
    '''Replaces (some known) ANSI coloring sequences to HTML

    For a more sophisticated solution, check out
    https://pypi.python.org/pypi/ansi2html
    '''
    return _RE_ANSICOLOR.sub(_ansicolors2html_repl, s)


def htmlspecialchars(s):
    '''Convert special characters to HTML entities
    '''
    return (s
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )

def text2html(s, whitespace=True):
    '''Convert plain text to HTML
    '''
    s = htmlspecialchars(s)

    if whitespace:
        s = s.replace(' ', '&nbsp;').replace('\n', '<br>')

    return ansicolors2html(s)


def _print_ANSI_SGR(code, text, **kwargs):
    if ANSI_SUPPORT:
        text = '\033[%sm%s\033[m' % (code, text)
    print(text, **kwargs)


def _make_print_fun(code):
    return lambda *a, **k: _print_ANSI_SGR(code, *a, **k)


error = _make_print_fun('31')  # red
warning = _make_print_fun('35')  # magenta
suggest = _make_print_fun('33')  # yellow
parrot = _make_print_fun('34')  # blue


def print_exc(strip_filenames=()):
    '''Print a colored traceback, with an (optionally) reduced stack
    '''
    import sys, traceback

    exc_type, exc_value, tb = sys.exc_info()

    for filename in strip_filenames:
        while tb and filename.startswith(tb.tb_frame.f_code.co_filename):
            tb = tb.tb_next

    s_list = traceback.format_exception(exc_type, exc_value, tb)
    error(''.join(s_list).rstrip())

    return exc_type, exc_value, tb
