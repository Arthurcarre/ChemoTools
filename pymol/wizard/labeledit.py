import sys
from pymol.wizard import Wizard
from pymol import cmd
from pymol import menu

class Labeledit(Wizard):

    def __init__(self, selection='', _self=cmd):
        Wizard.__init__(self, _self)

        self.selection_mode = cmd.get_setting_int("mouse_selection_mode")
        _self.set("mouse_selection_mode", 0)

        self.identifiers = None
        self.cursor = 0
        self.label = ""
        self.selename = _self.get_unused_name('_labeledit')
        self.sele = '?' + self.selename

        if selection:
            self.do_select(selection)

    def get_menu(self, tag):
        tmpmenu = {}

        for (a, text) in [
                ('color', 'Label Color'),
                ('bg_color', 'Background Color'),
                ]:
            m = tmpmenu[a] = [[ 2, text, '' ]]

            for color in ['default', 'front', 'back',]:
                m.append([ 1, color,
                    'cmd.get_wizard().set_' + a + '("' + color + '")'])

            m += [
                [ 0, '', '' ],
                [ 1, 'all colors', lambda s='label_' + a:
                    menu.by_rep_sub(self.cmd, 'labels', s, self.sele)],
            ]

        tmpmenu['size'] = [
            [ 2, 'Label Size', '' ]
        ] + [
            [ 1, str(size), 'cmd.get_wizard().set_size(%d)' % size]
            for size in (10, 14, 18, 24, 36)
        ]

        tmpmenu['bg_transparency'] = [
            [ 2, 'Background Transparency', '' ],
        ] + [
            [ 1, '%d%%' % (i), 'cmd.get_wizard().set_bg_transparency(%f)' % (i / 100.)]
            for i in (0, 20, 40, 60, 80)
        ]

        tmpmenu['connector'] = [
            [ 2, 'Connector', '' ],
            [ 1, 'Show', 'cmd.get_wizard().set_connector(1)' ],
            [ 1, 'Hide', 'cmd.get_wizard().set_connector(0)' ],
        ]

        return tmpmenu.get(tag, None)

    def cleanup(self):
        self.cmd.set("mouse_selection_mode", self.selection_mode)
        self.cmd.delete(self.selename)
        self.cmd.unpick()
        cmd.refresh_wizard()

    def get_event_mask(self):
        return \
            Wizard.event_mask_pick + \
            Wizard.event_mask_select + \
            Wizard.event_mask_special + \
            Wizard.event_mask_key

    def do_pick(self, bondFlag):
        self.do_select('?pk2' if self.cmd.count_atoms('?pk2') else '?pk1')

    def do_select(self, selection):
        n = self.cmd.select(self.selename, selection, 0)
        self.cmd.deselect()
        self.cmd.unpick()
        data = []

        if n == 0:
            print(" Warning: empty selection")
        elif n > 1:
            print(" Warning: selection must contain exactly one atom")
            self.cmd.select(self.selename, 'none', 0)
        else:
            self.cmd.edit(self.sele)
            self.cmd.iterate(self.sele, 'data[:] = [label,'
                             'model, segi, chain, resn, resi, name, alt,'
                             ']',
                             space={'data': data})

            self.label = data[0]
            self.cursor = len(self.label)

        self.identifiers = tuple(data[1:8])
        cmd.refresh_wizard()

    def do_special(self, k, x, y, mod):
        if k == 100:
            if self.cursor > 0:
                self.cursor -= 1
                cmd.refresh_wizard()
                return True
        elif k == 102:
            if self.cursor < len(self.label):
                self.cursor += 1
                cmd.refresh_wizard()
                return True
        return False

    def do_key(self, k, x, y, mod):
        pre, post = self.label[:self.cursor], self.label[self.cursor:]
        c = ''

        if k == 127:
            # delete
            post = post[1:]
        elif k == 8:
            # backspace
            if self.cursor > 0:
                pre = pre[:-1]
                self.cursor -= 1
        elif k == 27:
            # esc
            cmd.set_wizard()
            return True
        elif k in (10, 13):
            if not mod:
                cmd.set_wizard()
                return True
            # SHIFT+Return
            c = "\n"
        elif k < 32:
            return False
        elif k > 126:
            c = eval(r"u'\x%2x'" % (k))
        else:
            c = chr(k)

        if c:
            pre += c
            self.cursor += len(c)

        self.label = pre + post
        new_label = self.label

        self.cmd.alter(self.sele, 'label = new_label', space=locals())

        self.cmd.show('label', self.sele)

        cmd.refresh_wizard()
        return True

    def get_prompt(self):
        if not self.identifiers:
            self.prompt = [
                '\\669Pick any atom to edit its label...',
            ]
        else:
            i = self.cursor
            label = ''.join(('?' if c > '~' else c) for c in self.label)
            pre, c, post = label[:i], label[i:i+1], label[i+1:]
            if c in ('', ' '):
                c = '_'
            label = pre + '\\900' + c + '\\990' + post

            self.prompt = [
                '\\669Current atom: \\999/%s/%s/%s/%s`%s/%s`%s' % self.identifiers,
                '\\669Enter new label:\\990',
            ] + label.splitlines() + [
                '',
                '\\669(Shift-Return inserts newline)',
                '\\669(To edit different label, pick atom)',
                '',
                '\\669Use "3-Button Editing" CTRL-drag to move label',
            ]
        return self.prompt

    def set_nostate_setting(self, setting, value):
        self.cmd.alter_state(-1, self.sele, 's.' + setting + '= None')
        self.cmd.set(setting, value, self.sele)

    def set_color(self, color):
        self.cmd.set('label_color', color, self.sele)

    def set_bg_color(self, color):
        self.set_nostate_setting('label_bg_color', color)

    def set_bg_transparency(self, value):
        self.set_nostate_setting('label_bg_transparency', value)

    def set_connector(self, value):
        self.set_nostate_setting('label_connector', value)

    def set_size(self, size):
        self.cmd.set('label_size', size, self.identifiers[0])

    def get_panel(self):
        if self.identifiers:
            m = [
                [ 3, 'Label Color', 'color' ],
                [ 3, 'Background Color', 'bg_color' ],
                [ 3, 'Background Transparency', 'bg_transparency' ],
                [ 3, 'Connector', 'connector' ],
                [ 3, 'Label Size (object)', 'size' ],
            ]
        else:
            m = []

        return [
            [ 1, 'Editing Atom Label', '' ],
            ] + m + [
            [ 2, 'Done', 'cmd.set_wizard()' ]
            ]
