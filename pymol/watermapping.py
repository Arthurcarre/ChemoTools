# Presets for WaterMap results
# Copyright (c) Schrodinger, LLC.

cmd = __import__("sys").modules["pymol.cmd"]


def watermap_simple(selection='*', palette='green red', absolute=True,
        _self=cmd):
    '''
    Color by simple dG scheme.
    '''
    # AtomInfo.h: FLAGS 22-23 are for temporary use
    tempflag = 22
    tempflag_mask = 1 << tempflag

    count = [0]

    def callback(p, reps, vdw, flags):
        try:
            if p.r_watermap_free_energy is None:
                p.r_watermap_free_energy = (p.r_watermap_potential_energy +
                                            p.r_watermap_entropy + 19.67)
        except TypeError:
            return reps, vdw, (flags & ~tempflag_mask)
        count[0] += 1
        return (reps | 2), p.r_watermap_radius or 0.6, (flags | tempflag_mask)

    # set flag on atoms which have r_watermap_free_energy
    _self.alter(selection,
                '(reps, vdw, flags) = callback(p, reps, vdw, flags)',
                space={'callback': callback})

    if count[0] == 0:
        print('No atoms with "r_watermap_free_energy" property')
        return False

    # absolute or relative color scale
    minimum, maximum = (-2.5, 7.5) if int(absolute) else (None, None)

    # update colors for flagged atoms
    _self.spectrum('p.r_watermap_free_energy',
                   palette,
                   '({}) & flag {}'.format(selection, tempflag),
                   minimum=minimum,
                   maximum=maximum)

    _self.rebuild(selection)
    return True


def watermap_complex(selection='*', _self=cmd):
    '''
    Color by "complex" dH and -TdS scheme, set vdw=0.6 and show spheres.
    '''
    count = [0]
    clamp01 = lambda v: min(1.0, max(0.0, v))

    def callback(p, reps, vdw, color):
        try:
            enthalpyr = clamp01((p.r_watermap_potential_energy + 22.17) / 5.0)
            entropyr = clamp01(p.r_watermap_entropy / 5.0)
        except TypeError:
            return reps, vdw, color

        r, g, b = 0.0, (1.0 - 0.3 * (enthalpyr + entropyr)), 0.0
        if enthalpyr <= entropyr:
            b = 1.0 - enthalpyr / max(entropyr, 1e-100)
        else:
            r = 1.0 - entropyr / max(enthalpyr, 1e-100)
        color = (0x40000000
                 | (int(r * 0xFF) << 16)
                 | (int(g * 0xFF) << 8)
                 | (int(b * 0xFF)))
        count[0] += 1
        return (reps | 2), p.r_watermap_radius or 0.6, color

    _self.alter(selection,
                '(reps, vdw, color) = callback(p, reps, vdw, color)',
                space={'callback': callback})

    if count[0] == 0:
        print('No atoms with "r_watermap_potential_energy" property')
        return False

    _self.rebuild(selection)
    return True


def watermap_predefined(selection='*', quiet=0, _self=cmd):
    '''
    Color by r_watermap_color_r/g/b, set vdw to r_watermap_radius, and
    show spheres.
    '''
    count = [0]

    def callback(p, reps, vdw, color):
        try:
            color = (0x40000000
                     | (int(p.r_watermap_color_r * 0xFF) << 16)
                     | (int(p.r_watermap_color_g * 0xFF) << 8)
                     | (int(p.r_watermap_color_b * 0xFF)))
        except TypeError:
            return reps, vdw, color
        count[0] += 1
        return (reps | 2, p.r_watermap_radius or 0.6, color)

    _self.alter(selection,
                '(reps, vdw, color) = callback(p, reps, vdw, color)',
                space={'callback': callback})

    if count[0] == 0:
        if not int(quiet):
            print('No atoms with "r_watermap_color_r/g/b" properties')
        return False

    _self.rebuild(selection)
    return True
