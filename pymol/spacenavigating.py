import sys
import time
import threading
import pymol

SCALE = 0.01

instance = None


class SpaceNavStartError(Exception):
    pass


def _fix_spnav_py3():
    # spnav module uses PyCObject_AsVoidPtr, which is not available in Python 3

    from ctypes import pythonapi

    try:
        pythonapi.PyCObject_AsVoidPtr
    except:
        # dummy object sufficient for us
        pythonapi.PyCObject_AsVoidPtr = type('', (), {})()


def start(*args, quiet=0, **kwargs):
    '''Start a global SpaceNavControl instance
    '''
    global instance

    if instance is not None:
        stop()

    try:
        instance = SpaceNavControl(*args, **kwargs)
        instance.start()
        return True
    except SpaceNavStartError as e:
        if not quiet:
            print(e)
    except ImportError as e:
        if not quiet:
            print(e)
            pymol.parser.print_install_suggestion(e)
    except BaseException as e:
        print(e)

    instance = None
    return False


def stop():
    if instance is None:
        return

    try:
        instance.stop()
    except BaseException as e:
        print(e)


class SpaceNavControl:
    '''Handles 3Dconnexion SpaceNavigator on Windows and Linux.

    On MacOS, PyMOL uses the C-level SDK.
    '''

    def __init__(self, focus_test=None, _self=pymol.cmd):
        self.cmd = _self

        if focus_test is None:
            focus_test = lambda: True
        self.focus_test = focus_test

    def start(self):
        if pymol.IS_WINDOWS:
            self._start_spacenavigator()
        elif pymol.IS_LINUX:
            self._start_spnav()
        elif pymol._cmd._3dconnexion_init():
            self.stop = pymol._cmd._3dconnexion_shutdown
        else:
            raise SpaceNavStartError('3DxWare driver not available')

    def _start_spnav(self):
        _fix_spnav_py3()
        import spnav

        try:
            spnav.spnav_open()
        except spnav.SpnavConnectionException as e:
            raise SpaceNavStartError(str(e))

        self._running = True
        self.stop = lambda: setattr(self, '_running', False)

        t = threading.Thread(target=self._spnav_event_loop)
        t.daemon = True
        t.start()

    def _spnav_event_loop(self):
        import spnav

        kill_pending_events = False

        try:
            while self._running:
                if not self.focus_test():
                    kill_pending_events = True
                    self.cmd._sdof(0., 0., 0., 0., 0., 0.)
                    time.sleep(0.1)
                    continue

                if kill_pending_events:
                    kill_pending_events = False
                    while spnav.spnav_poll_event() is not None:
                        pass

                event = spnav.spnav_wait_event()

                if event.ev_type == spnav.SPNAV_EVENT_MOTION:
                    t, r = event.translation, event.rotation
                    args = (t[0], -t[2], -t[1], r[0], -r[2], -r[1])
                    args = [v * SCALE for v in args]
                    self.cmd._sdof(*args)
                elif event.ev_type == spnav.SPNAV_EVENT_BUTTON:
                    if event.press:
                        self.cmd._sdof_button(event.bnum + 1)
        except spnav.SpnavWaitException:
            print('SpnavWaitException: spacenavd stopped?')
        finally:
            spnav.spnav_close()

    def _start_spacenavigator(self):
        import spacenavigator
        device = spacenavigator.open(self._spacenavigator_callback,
                                     self._spacenavigator_button_callback)
        if device is None:
            raise SpaceNavStartError('failed to open spacenavigator device')

        # restore 0.1.5 behavior
        device.device.set_raw_data_handler(lambda x: device.process(x))

        self.stop = device.close

    def _spacenavigator_callback(self, s):
        if self.focus_test():
            args = (s[1], -s[2], -s[3], -s[5], -s[4], s[6])
            args = [v * SCALE * 350 for v in args]
        else:
            args = (0., 0., 0., 0., 0., 0.)
        self.cmd._sdof(*args)

    def _spacenavigator_button_callback(self, state, button):
        if self.focus_test():
            if isinstance(button, list):
                button = sum((b << i) for (i, b) in enumerate(button))
            self.cmd._sdof_button(button)
