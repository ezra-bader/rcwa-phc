import S4
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import threading
import time, warnings, itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import interp1d, griddata
from pathlib import Path
from joblib import Parallel, delayed
import re as _re
from datetime import datetime as _dt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import filedialog
import tempfile
import os
import shutil
import dataclasses
from dataclasses import dataclass, field, asdict
import json
from importlib import reload
warnings.filterwarnings('ignore')

# region Filename generation, tensor rotation helpers, material class definitions, layer class definitions

# Need to save layer stack in some file names, but some characters are problematic. 
# This builds a short, filesystem-safe slug that fully identifies the layer stack.
def make_stack_slug(conf: object) -> str:
    layers = conf.layers
    a      = conf.lattice_const

    def fmt(x):
        # Auto-scale: use µm if >= 1000nm to keep filenames short
        if x >= 1000:
            s = f'{x/1000:.2f}'.rstrip('0').rstrip('.')
            return s + 'um'
        s = f'{float(x):.1f}'.rstrip('0').rstrip('.')
        return s + 'nm'

    # Build entry list, skipping Air padding
    entries = []
    for i, L in enumerate(layers):
        mat = L.material if isinstance(L.material, str) else L.get_material().name
        is_first = (i == 0)
        is_last  = (i == len(layers) - 1)
        if mat == 'Air' and L.pattern is None and (is_first or is_last):
            continue
        entries.append((mat, L.thickness, L.pattern, L.ff,
                         L.layer_rot, L))

    # Collapse repeating 2-layer DBR pairs
    parts = [f'A{fmt(a)}']
    i = 0
    while i < len(entries):
        mat, thick, pat, ff, rot, layer_obj = entries[i]

        # Try to match a repeating 2-layer unit
        if i + 1 < len(entries):
            unit = (entries[i][0], entries[i+1][0])
            count = 1
            j = i + 2
            while (j + 1 < len(entries) and
                   entries[j][0]   == unit[0] and
                   entries[j+1][0] == unit[1] and
                   entries[j][1]   == entries[i][1] and    # same thickness
                   entries[j+1][1] == entries[i+1][1]):
                count += 1
                j += 2
            if count >= 2:
                m1, t1 = entries[i][0],   entries[i][1]
                m2, t2 = entries[i+1][0], entries[i+1][1]
                parts.append(f'({m1}{fmt(t1)}-{m2}{fmt(t2)})x{count}')
                i = j
                continue

        # Single layer
        seg = f'{mat}-{fmt(thick)}'
        if pat in ('hole', 'pillar'):
            seg += f'-{pat}-ff{ff:.2f}'.rstrip('0').rstrip('.')
        elif pat == 'cuboids':
            seg += f'-cub-w0{fmt(layer_obj.w0)}-a{layer_obj.alpha:.2f}'.rstrip('0').rstrip('.')
        if rot:
            seg += f'_rot{int(round(rot)):.2f}'.rstrip('0').rstrip('.')
        parts.append(seg)
        i += 1

    return '_'.join(parts)

# We don't want to bother writing out a full filename every time we run a study.
# Instead, we construct automatically from the study config.
def make_study_fname(study: int, conf: object) -> str:
    """
    Generate a descriptive, collision-resistant filename for simulation output.

    Parameters
    ----------
    study      : int — 0, 1, 2, or 3
    conf       : object - RCWAConfig used to run the study

    Returns
    -------
    Study filename with path prepended. The parent directory is created if it doesn't exist.
    If there is a clashing file in the save directory, the new one is saved as filename_v2, _v3, etc.

    Examples
    --------
    study0_P500_SiO2-50_hole-ff0.5_CrSBr-70_hole-ff0.5_lam400-800.csv
    study1_P500_CrSBr-70_hole-ff0.5_rot00.csv
    study2_P500_ReS2-70_cuboids-w0250-a0.6_rot00.csv
    study3_P500_CrSBr-70_hole-ff0.5_azim0_lam885-1033.csv
    """
    output_dir = conf.output_dir
    layers = conf.layers
    a = conf.lattice_const
    n_basis = conf.n_basis

    slug   = make_stack_slug(conf)
    parts  = [f'study{study}', slug]

    # We only want to specify global tensor index rotation if non-zero value specified in config:
    if conf.global_rot is not None: parts.append(f'rot{int(round(conf.global_rot)):02d}')

    parts.append(f'nb{n_basis}')

    fname = '_'.join(parts)

    out   = Path(output_dir) / fname

    # Collision guard: if exact name exists, append _v2, _v3, ...
    if (Path(output_dir) / (fname + '.csv')).exists():
        stem = fname
        v = 2
        while True:
            candidate = stem + f'_v{v}'
            if not (Path(output_dir) / (candidate + '.csv')).exists():
                fname = candidate
                break
            v += 1
    return fname

# Once we have our dataframe from a study, we want to save it
# alongside a .json of the same name containing all the config information to be used
# (a) when plotting or (b) to rebuild a study config at a later date.
# Returns path_to_csv, path_to_json.
def save_study(df, conf: object, fname_stem: str):
    '''Save study DataFrame as CSV and conf as JSON, both named fname_stem -->
    the plain string from make_study_fname (no extension or directory).
    Returns (csv_path, json_path).'''
    base = conf.output_dir
    csv_path = base / (fname_stem + '.csv')
    json_path = base / (fname_stem + '.json')
    df.to_csv(csv_path, index=False)
    conf.save(json_path)
    return csv_path, json_path

# —— Tensor rotation helpers —————————————————————————————————————————————————

# If we want to rotate our tensor index in-plane, i.e.
# counter-clockwise about the z-axis. 0° means n_aa still points along x-axis;
# 90° means n_aa now points along y-axis, etc.
def rotation_matrix_z(rot_deg: float) -> np.ndarray:
    '''Rotation about z-axis by phiz_deg.'''
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rotate_epsilon_tensor(ea, eb, ec,
                          rot_deg: float = 0.0,
                          ) -> tuple:
    '''
    rot_deg:              in-plane rotation of index tensor about ẑ
    '''
    eps_diag = np.diag([ea, eb, ec]).astype(complex)
    R = rotation_matrix_z(rot_deg)
    eps_rot = R @ eps_diag @ R.T
    return tuple(tuple(row) for row in eps_rot)

# —— Base class ———————————————————————————————————————————————————————————————

# Empty class for generic material:
class Material(ABC):
    '''
    Abstract base class for all materials. Each subclass implements:
        epsilon(lam_nm, layer_tilt, layer_rot) -> 3x3 tuple
    '''
    name: str = ''

    @abstractmethod
    def epsilon(self, lam_nm: float, 
                rot: float = 0.0,
                ) -> tuple:
        ...
    def __rep__(self):
        return f"{self.__class__.__name__}({self.name})"

# —— Isotropic, non-dispersive materials ——————————————————————————————————————
# A simple material with the same index of refraction in all directions.
# Can initialize with n = float or eps = float. Prioritizes epsilon.
class IsotropicMaterial(Material):
    '''
    Isotropic, non-dispersive material. Rotation angles have no effect.
    '''
    def __init__(self, name: str, n: float = None, eps: float = None):
        self.name = name
        if not (eps or n): raise ValueError("Provide either n: float = index or eps: float = permittivity.")
        if eps:
            self._eps = complex(eps)
            self._n   = complex(eps**(1/2))
        elif n:
            self._eps = complex(n**2)
            self._n = complex(n)

    # Other kwargs included for consistency/clarity,
    # but will obviously have no effect on a fully isotropic material.
    # Returns epsilon tensor to pass to S4.
    def epsilon(self, lam_nm: float,
                rot: float = 0.0,
                ) -> tuple:
        e = self._eps
        return ((e,0,0),(0,e,0),(0,0,e))
    
# —— Anisotropic, non-dispersive materials ——————————————————————————————————————
# Directly supply index along each principal axis as a float.
# Also takes complex epsilon_ii.
class AnisotropicMaterial(Material):
    '''
    Anisotropic, non-dispersive material. Directly supply each index.
    Ex: ReS2 (without exciton.)
    '''
    def __init__(self, name: str, 
                 n_a: float = None, k_a: float = 0.0,
                 n_b: float = None, k_b: float = 0.0,
                 n_c: float = None, k_c: float = 0.0,
                 eps_xx: complex = None,
                 eps_yy: complex = None,
                 eps_zz: complex = None):
        self.name = name
        match eps_xx:
            case None:
                match n_a:
                    case None:
                        raise ValueError("Provide either n_a or eps_aa.")
                    case _:
                        self._eps_xx = (n_a + 1j*k_a)**2
            case _:
                self._eps_xx = eps_xx
        match eps_yy:
            case None:
                match n_b:
                    case None:
                        raise ValueError("Provide either n_b or eps_bb.")
                    case _:
                        self._eps_yy = (n_b + 1j*k_b)**2
            case _:
                self._eps_yy = eps_yy
        match eps_zz:
            case None:
                match n_c:
                    case None:
                        raise ValueError("Provide either n_c or eps_zz.")
                    case _:
                        self._eps_zz = (n_c + 1j*k_c)**2
            case _:
                self._eps_zz = eps_zz

    def epsilon(self, lam_nm: float,
                rot:        float = 0.0,
                ) -> tuple:
        return rotate_epsilon_tensor(self._eps_xx, self._eps_yy, self._eps_zz,
                                     rot)

# —— Birefringent, non-dispersive materials (very similar to anisotropic, but optimized for ReS2)
class BirefringentMaterial(Material):
    '''
    Birefringent, non-dispersive material. Supply eps0, delta_eps, and eps_zz.
    Ex: ReS2 (without exciton.)
    '''
    def __init__(self, name: str, 
                 eps0: float = None,
                 delta_eps: float = None,
                 eps_zz: float = None):
        self.name = name
        self._eps_xx = eps0 + (delta_eps)/2
        self._eps_yy = eps0 - (delta_eps)/2
        self._eps_zz = eps_zz

    def epsilon(self, lam_nm    : float,
                rot             : float = 0.0,
                )-> tuple:
        return rotate_epsilon_tensor(self._eps_xx, self._eps_yy, self._eps_zz,
                                     rot)

# —— Uniaxial, non-dispersive materials ———————————————————————————————————————
# One extraordinary axis, two ordinary.
# All polarizations traveling along the extraordinary axis will not experience birefringence.
# E.g. liquid crystal.
class UniaxialMaterial(Material):
    '''
    Non-dispersive uniaxial material.
    n_e          : extraordinary index (along the optical axis)
    n_o          : ordinary index (along other 2 axes)
    optical_axis : 'x', 'y', or 'z' — which lab axis to set the material's optical axis along prior to any index rotation
    Example — homeotropic nematic liquid crystal (long molecular axis, which is parallel with optical axis, aligned normal to substrate):
        UniaxialMaterial('LC', n_o = 1.50, n_e = 1.91, optical_axis = 'z')
        '''
    
    x, y, z = np.eye(3)
    _AXIS_MAP = {
        'x': x,
        'y': y,
        'z': z
    }

    def __init__(self, name: str,
                 n_e: float, n_o: float, optical_axis: str = 'z',
                 k_e: float = 0.0, k_o: float = 0.0): # if you want extinction, otherwise just 0
        if optical_axis not in self._AXIS_MAP:
            raise ValueError(f"optical_axis must be 'x', 'y', or 'z', got '{optical_axis}'")
        self.name = name
        self.optical_axis = optical_axis
        self._n_e = n_e; self._k_e = k_e
        self._n_o = n_o; self._k_o = k_o

        # Want to build diagonal epsilon with n_e along chosen axis
        eps_e = (n_e + 1j*k_e)**2
        eps_o = (n_o + 1j*k_o)**2
        diag = {'x': (eps_e, eps_o, eps_o),
                'y': (eps_o, eps_e, eps_o),
                'z': (eps_o, eps_o, eps_e)}[optical_axis]
        self._ea, self._eb, self._ec = diag

    def epsilon(self, lam_nm: float,
                rot: float = 0.0,
                ) -> tuple:
        return rotate_epsilon_tensor(self._ea, self._eb, self._ec,
                                     rot)
    
    def __repr__(self):
        return(f"UniaxialMaterial('{self.name}', n_o={self._n_o}, n_e={self._n_e}, "
               f"optical_axis='{self.optical_axis}')")
    
# —— Dispersive, uniaxial materials (b axis from file, a & c constant) ————————————————
class DispersiveMaterial(Material):
    '''Material with one dispersive index. Optimized for CrSBr. File columns: lambda(nm), n_b, n_k.
    Background a and c are constant (n_a, n_c).
    '''
    def __init__(self, name: str, nk_file: str, n_a: float = 3.0, n_c: float = 3.0,
                    k_a: float = 0.0, k_c: float = 0.0,
                    delimiter: str = None, skip_header: int = 0,
                    reverse: bool = False):
        self.name = name
        self._n_a = n_a
        self._n_c = n_c
        self._k_a = k_a
        self._k_c = k_c
        data = np.loadtxt(Path(nk_file), delimiter = delimiter,
                            skiprows = skip_header)
        if reverse:
            data = data[::-1]
        lam = data[:,0]
        self._nb = interp1d(lam, data[:,1], kind='cubic', fill_value='extrapolate')
        self._kb = interp1d(lam, data[:,2], kind='cubic', fill_value='extrapolate')
    def epsilon(self, lam_nm: float,
                rot: float = 0.0,
                ) -> tuple:
        ea: complex = (self._n_a + 1j*self._k_a)**2
        eb: complex = (float(self._nb(lam_nm)) + 1j*float(self._kb(lam_nm)))**2
        ec: complex = (self._n_c + 1j*self._k_c)**2
        return rotate_epsilon_tensor(ea, eb, ec,
                                        rot)
        
# —— Dispersive biaxial (all 3 axes from file) ————————————————————————————————
class DispersiveBiaxialMaterial(Material):
    '''Dispersive, fully biaxial. File columns: lambda, n_xx, k_xx, n_yy, k_yy, n_zz, k_zz.
    Example: MoOCl2.'''
    def __init__(self, name: str, nk_file: str,
                 delimiter: str = ',', skip_header: int = 0,
                 reverse: bool = False):
        self.name = name
        data = np.loadtxt(Path(nk_file), delimiter = delimiter,
                          skiprows = skip_header)
        if reverse:
            data = data[::-1]
        lam = data[:,0]
        self._na = interp1d(lam, data[:,1], kind='cubic', fill_value='extrapolate')
        self._ka = interp1d(lam, data[:,2], kind='cubic', fill_value='extrapolate')
        self._nb = interp1d(lam, data[:,3], kind='cubic', fill_value='extrapolate')
        self._kb = interp1d(lam, data[:,4], kind='cubic', fill_value='extrapolate')
        self._nc = interp1d(lam, data[:,5], kind='cubic', fill_value='extrapolate')
        self._kc = interp1d(lam, data[:,6], kind='cubic', fill_value='extrapolate')
    def epsilon(self, lam_nm: float,
                rot: float = 0.0,
                ) -> tuple:
        ea = (float(self._na(lam_nm)) + 1j*float(self._ka(lam_nm)))**2
        eb = (float(self._nb(lam_nm)) + 1j*float(self._kb(lam_nm)))**2
        ec = (float(self._nc(lam_nm)) + 1j*float(self._kc(lam_nm)))**2
        return rotate_epsilon_tensor(ea, eb, ec,
                                     rot)
    
# —— Dispersive isotropic (nk from file) ————————————————————————————————
class DispersiveIsotropicMaterial(Material):
    '''Dispersive, isotropic. File columns: lambda, n, k.
    Example: SMILES.'''
    def __init__(self, name: str, nk_file: str,
                 delimiter: str = ',', skip_header: int = 0,
                 reverse: bool = False):
        self.name = name
        data = np.loadtxt(Path(nk_file), delimiter = delimiter,
                          skiprows = skip_header)
        if reverse:
            data = data[::-1]
        lam = data[:,0]
        ncol   = data[:,1]
        kcol   = data[:,2]
        ninterp = interp1d(lam, ncol, kind='cubic', bounds_error=False,fill_value=(ncol[0],ncol[-1]))
        kinterp = interp1d(lam, kcol, kind='cubic', bounds_error=False,fill_value=(kcol[0],kcol[-1]))
        self._na = ninterp
        self._ka = kinterp
        self._nb = ninterp
        self._kb = kinterp
        self._nc = ninterp
        self._kc = kinterp
    def epsilon(self, lam_nm: float,
                rot: float = 0.0,
                ) -> tuple:
        ea = (float(self._na(lam_nm)) + 1j*float(self._ka(lam_nm)))**2
        eb = (float(self._nb(lam_nm)) + 1j*float(self._kb(lam_nm)))**2
        ec = (float(self._nc(lam_nm)) + 1j*float(self._kc(lam_nm)))**2
        return rotate_epsilon_tensor(ea, eb, ec,
                                     rot)
    
class Layer:
    '''
    One layer in the stack.
    Parameters
    ----------------
    material            : str (key in RU_MATERIALS) or Material instance
    thickness           : float, nm
    pattern             : None | 'hole' | 'pillar' | 'cuboids'
    ff                  : fill factor override (defaults to global FF)
    layer_rot           : in-plane index tensor rotation (CCW)
    '''
    def __init__(self, material, thickness: float,
                 pattern = None, ff: float = None, 
                 layer_rot: float = None,
                 w0: float = None,
                 alpha: float = None):
        self.material           = material
        self.thickness          = thickness
        self.pattern            = pattern
        self.ff                 = ff
        self.layer_rot          = layer_rot
        self.w0                 = w0
        self.alpha              = alpha
        if self.pattern == 'cuboids' and not (self.w0 or self.alpha):
            raise ValueError("Please specify w0 and alpha for the cuboid pattern!")
        if (self.pattern == 'hole' or self.pattern == 'pillar') and not self.ff:
            raise ValueError("Please specify a fill factor for any hole or pillar layers!")
    def get_cuboid_geometry(self, a: float):
        """
        Compute cuboid pillar positions and sizes for this layer.
        Uses layer-level w0/alpha if specified, otherwise falls back to globals.
        Returns (W1, W2, X1, Y1, X2, Y2).
        """
        w0    = self.w0 
        alpha = self.alpha
        w1 = w0 * np.sqrt(2 / (1 + alpha**2))
        w2 = w1 * alpha
        g  = (a - w1 - w2) / 3
        x1 = g + w1/2;       y1 = g + w1/2
        x2 = a - g - w2/2;   y2 = a - g - w2/2
        return w1, w2, x1, y1, x2, y2
    def get_material(self) -> Material:
        if isinstance(self.material, Material):
            return self.material
        return RU_MATERIALS[self.material]
    def epsilon(self, lam_nm: float) -> tuple:
        return self.get_material().epsilon(
            lam_nm,
            rot       = self.layer_rot or 0.0,
        )
    def __repr__(self):
        match self.pattern:
            case 'hole' | 'pillar':
                pat = f', {self.pattern}, ff={self.ff}'
            case 'cuboids':
                pat = f', {self.pattern}, w0={self.w0}, α={self.alpha}'
            case None:
                pat = ''
            case _:
                raise ValueError("Unknown pattern entered.")
        match self.layer_rot:
            case None | 0.0 | 0:
                rot = ''
            case _:
                rot = f', rot={self.layer_rot:.0f}°'
        return f"Layer({self.material}, {self.thickness:.2f}nm{pat}{rot})"

#endregion

# —— Material library —————————————————————————————————————————————————————————
RU_MATERIALS = {
        'Air': IsotropicMaterial('Air', n=1.0),
        'Si': IsotropicMaterial('Si', n=3.4),
        'SiO2': IsotropicMaterial('SiO2', n=1.46),
        'TiO2': IsotropicMaterial('TiO2', n=2.2),
        'SiN': IsotropicMaterial('SiN', n=2.05),
        'CrSBr': DispersiveMaterial('CrSBr', 'crsbr_nk_yinming.txt', 
                                            n_a = 3.0, n_c = 3.0),
        'MoOCl2': DispersiveBiaxialMaterial('MoOCl2',
                                            'moocl2_nm_nxx_kxx_nyy_kyy_nzz_kzz.csv',
                                            delimiter=',', reverse=True),
        'ReS2': BirefringentMaterial('ReS2', eps0=18.0, delta_eps=1.7, eps_zz=7.25),
        'LC': UniaxialMaterial('LC',n_o=1.50,n_e=1.91, optical_axis='x'),
        'SMILES': DispersiveIsotropicMaterial('SMILES',nk_file='SMILES_NK.csv',reverse=False),
        # 'WS2': DispersiveIsotropicMaterial('WS2', nk_file='ws2_exciton620nm.csv')
        #To be added: liquid crystal with tilt
        #'LC': UniaxialMaterial('LC', n_o=1.50, n_e=1.70)
        #To be added: DBR layer components
        #''
    }
def load_material_library():
    mat_keys_str = ", ".join(RU_MATERIALS)
    print(f'Material library loaded. Currently supported materials:')
    print(mat_keys_str)
    return RU_MATERIALS

# Simple function to build DBR stack (DBR, air, DBR)
def DBR(n_pairs: int,
        high_index_mat: str = 'TiO2',
        low_index_mat:  str = 'SiO2',
        centerlam_nm:   float = 550,
        orientation: str = 'top') -> list:
    """
    Returns a list of quarter-wave Layer pairs for one DBR mirror.

    reversed=False (default): starts H, ends L  --> use for bottom mirror
    reversed=True:            starts L, ends H  --> use for top mirror

    Use with * unpacking:
        MY_LAYERS = [
            ru.Layer('Air', 500),
            *ru.DBR(6, centerlam_nm=550, reversed=True),   # top: L...H | cavity
            ru.Layer('LC', 3450, layer_tilt=51.15),
            *ru.DBR(6, centerlam_nm=550),                  # bottom: H...L
            ru.Layer('Air', 500),
        ]
    """
    def _n(mat_name):
        return float(np.sqrt(RU_MATERIALS[mat_name]._eps).real)

    d_H = centerlam_nm / (4 * _n(high_index_mat))
    d_L = centerlam_nm / (4 * _n(low_index_mat))

    pair = [Layer(low_index_mat, d_L), Layer(high_index_mat, d_H)] if orientation == 'top' \
      else [Layer(high_index_mat, d_H), Layer(low_index_mat,  d_L)]

    return pair * n_pairs


def check_layers(layers: list = None, conf: object = None):
    if not (layers or conf):
        print(f"Please indicate layer stack to check.\nExample: check_layers(layers=LAYERS)")
        return
    if conf:
        layers = conf.layers

    # Column widths
    CW = {'i': 8, 'mat': 12, 'thick': 22, 'pat': 24, 'rot': 8}

    header = (f"{'layer':<{CW['i']}}"
              f"{'material':<{CW['mat']}}"
              f"{'thickness (nm)':<{CW['thick']}}"
              f"{'pattern':<{CW['pat']}}"
              f"{'rot (°)':<{CW['rot']}}"
              )
    divider = '-' * sum(CW.values())

    lines = [header, divider]

    for i, L in enumerate(layers):
        # Format thickness to 2 decimal places
        thick = f'{L.thickness:.2f}'.rstrip('0').rstrip('.')

        # Format pattern
        match L.pattern:
            case None:
                pat = 'None'
            case 'hole' | 'pillar':
                pat = f'{L.pattern}, ff={L.ff}'
            case 'cuboids':
                pat = f'cuboids, w0={L.w0}, a={L.alpha}'

        # Format rot
        match L.layer_rot:
            case 0.0:
                rot = 0
            case None:
                if conf:
                    rot = f'{conf.global_rot:.1f}' if conf.global_rot else 0
                else: rot = 0
            case _:
                rot = f'{L.layer_rot:.1f}'

        lines.append(f"{f'{i}':<{CW['i']}}"
                     f"{str(L.material):<{CW['mat']}}"
                     f"{thick:<{CW['thick']}}"
                     f"{pat:<{CW['pat']}}"
                     f"{rot:<{CW['rot']}}"
                     )

    print('\n'.join(lines))


# -- Making angle values, pairs for angle-resolve studies

# ── Elevation / azimuth grids ──────────────────────────────────────────────
def make_elev_vals(elev_max_deg, n_pts):
    """Elevation angles evenly spaced in sin(elev) — avoids elev=0."""
    sin_max  = np.sin(np.radians(elev_max_deg))
    sin_vals = np.linspace(-sin_max, sin_max, n_pts)
    return np.round(np.degrees(np.arcsin(sin_vals)),decimals=3)

def make_kspace_grid(elev_max_deg, n_elev, n_azim_max, n_azim_min=12, start_frac=0.02):
    """
    Hybrid k-space grid: uniform density in k-space area,
    with a minimum azimuthal count so the BIC region near Γ is well-sampled.
    Returns list_of_(elev), list_of_(azim,elev)_pairs.
    """
    sin_max   = np.sin(np.radians(elev_max_deg))
    sin_vals  = np.linspace(start_frac * sin_max, sin_max, n_elev)
    elev_vals = np.degrees(np.arcsin(sin_vals))
    pairs = []
    for sin_e, elev in zip(sin_vals, elev_vals):
        n_az = max(n_azim_min, round(n_azim_max * sin_e / sin_max))
        for az in np.linspace(0, 360, n_az, endpoint=False):
            pairs.append((az, elev))
    return elev_vals, pairs

@dataclass
class Study1Config:
    elev_max: float = 35.0
    elev_n: int = 70
    azim_vals: list = field(default_factory=list)
    
    @property
    def elev_vals(self):
        return make_elev_vals(self.elev_max, self.elev_n)
    def elev_vals_coarse(self):
        return make_elev_vals(self.elev_max, self.elev_n//2)

@dataclass
class Study2Config:
    elev_max: float = 40.0
    elev_n: int = 40
    azim_n_max: int = 180
    azim_n_min: int = 45
    
    @property
    def elev_vals(self):
        return make_kspace_grid(self.elev_max,self.elev_n,self.azim_n_max,self.azim_n_min)[0]
    @property
    def pairs(self):
        return make_kspace_grid(self.elev_max,self.elev_n,self.azim_n_max,self.azim_n_min)[1]

@dataclass
class RCWAConfig:
    # Wavelengths
    lam_start: float
    lam_stop: float
    lam_step: float

    # Global index rotations
    global_rot          : float = None      # spins tensor index around z axis

    # Lattice constant
    lattice_const: float = None

    # Material stack
    layers: list = field(default_factory=list)
    def stack_summary(self):
        return check_layers(layers=self.layers)
    
    # Solver settings
    n_basis: int = 100
    n_grid: int = 10
    n_jobs: int = 10

    # Study objects
    study1: Study1Config = None
    study2: Study2Config = None

    # File saving
    save_to: str = None
    save_figs: bool = False

    @property
    def wavelengths(self):
        return np.arange(self.lam_start, self.lam_stop + self.lam_step/2, self.lam_step)
    
    @property
    def output_dir(self):
        match self.save_to:
            case None:
                print(f'\nWarning: no output directory selected.'
                      f'\nFiles will be saved to folder "rcwa_output" in current directory ({Path.cwd() / "rcwa_output"}).'
                      f'\nTo set an output path, specify RCWAConfig.save_to = "/your_path_here".'
                      f"\nTo use the default output directory without seeing this warning, specify RCWAConfig.save_to = 'default'.")
                savepath = Path.cwd() / 'rcwa_output'
                savepath.mkdir(parents=True, exist_ok = True)
                return savepath
            case 'default':
                savepath = Path.cwd() / 'rcwa_output'
                savepath.mkdir(parents=True, exist_ok = True)
                return savepath
            case _:
                savepath = Path(self.save_to)
                savepath.mkdir(parents=True, exist_ok=True)
                return savepath
    
    def verify_config(self):
        check_layers(layers = None, conf = self)
        print(f"\nWavelengths: {self.wavelengths[0]:.0f}–{self.wavelengths[-1]:.0f} nm ({len(self.wavelengths)} pts)")
        print(f"Study 1: {len(self.study1.elev_vals)} elev × {len(self.study1.azim_vals)} azim = {len(self.study1.elev_vals)*len(self.study1.azim_vals)} pairs")
        print(f"Study 2: {len(self.study2.pairs)} (azim, elev) pairs")
        print(f"Workers: {self.n_jobs}  |  N_BASIS: {self.n_basis}")
        print(f'\nOutput directory: {self.output_dir}')
    
    def save(self, path: Path):
        '''
        Saves config metadata to path.with_suffix('.json').
        Picks up new RCWAConfig field automatically, if any were added.
        Layers are serialized via their own __dict__.
        '''
        def serialize_value(v):
            if isinstance(v, list) and v and type(v[0]).__name__ == 'Layer':
                return [
                    {k: (val if isinstance(val, (int, float, str, bool, type(None)))
                        else val.name if isinstance(val, Material) else str(val))
                    for k, val in L.__dict__.items()
                    if not k.startswith('_')}
                    for L in v
                ]
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return dataclasses.asdict(v)
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        #convert dataclass to dictionary
        config_dict = {
            f.name: serialize_value(getattr(self, f.name))
            for f in dataclasses.fields(self)
        }
        out = Path(path)
        with open(out, 'w') as f:
            json.dump(config_dict, f, indent=4, check_circular=False)
        print(f'{out.parent}\n|\n|\nV')
        print(f"Config saved as {out.name}")

    @classmethod
    def load(cls, json_path) -> object:
        '''Reconstructs the config object from a JSON file.'''
        path = Path(json_path)
        if path.suffix != '.json':
            path = Path(str(json_path) + '.json')
        
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle nested study configs
        if data.get('study1') is not None:
            data['study1'] = Study1Config(**data['study1'])
        if data.get('study2') is not None:
            data['study2'] = Study2Config(**data['study2'])
        if data.get('layers') is not None:
            data['layers'] = [Layer(**d) for d in data['layers']]
        
        return cls(**data)
            


# -- Plotting unit cell, refractive indices
RU_MATERIAL_COLORS = {
    'Air':    ('lightcyan',      'gray',          0.12),
    'SiO2':   ('paleturquoise',  'steelblue',     0.35),
    'TiO2':   ('indianred',      'brown',        0.35),
    'SiN':    ('slategray',      'navy',          0.75),
    'Si':     ('darkgray',       'black',         0.60),
    'CrSBr':  ('goldenrod',      'saddlebrown', 0.60),
    'MoOCl2': ('darkgoldenrod',  'saddlebrown',   0.70),
    'ReS2':   ('mediumpurple',   'indigo',        0.70),
    'LC':     ('bisque',      'orange',     0.60),
    'SMILES': ('hotpink',   'deeppink',      0.60)
}
def get_material_colors():
    return RU_MATERIAL_COLORS

def _mat_color(material_name):
    """Return (facecolor, edgecolor, alpha) for a material name."""
    return RU_MATERIAL_COLORS.get(material_name, ('wheat', 'k', 0.5))
 
 # ── Refractive index plot ─────────────────────────────────────────────────────
def plot_refractive_index(conf: object, layers=None, materials=None,
                           lam_min=None, lam_max=None, n_pts=600):
    """
    Plot n and k for dispersive materials.
 
    Parameters
    ----------
    layers    : list of Layer -- auto-detect dispersive materials from stack.
    materials : list of str  -- override: plot only these material names.
                e.g. materials=['CrSBr']
    lam_min, lam_max : float -- wavelength range (nm).
                Defaults to LAM_START / LAM_STOP if not specified.
    """
    if lam_min is None:
        lam_min = conf.lam_start
    if lam_max is None:
        lam_max = conf.lam_stop
    lam_plot = np.linspace(lam_min, lam_max, n_pts)
 
    if materials is not None:
        to_plot = [RU_MATERIALS[m] for m in materials if m in RU_MATERIALS]
    elif layers is not None:
        seen = {}
        for layer in layers:
            mat = layer.get_material()
            if mat.name not in seen and not isinstance(mat, IsotropicMaterial):
                seen[mat.name] = mat
        to_plot = list(seen.values())
    else:
        layers = conf.layers
        seen = {}
        for layer in layers:
            mat = layer.get_material()
            if mat.name not in seen and not isinstance(mat, IsotropicMaterial):
                seen[mat.name] = mat
        to_plot = list(seen.values())
 
    if not to_plot:
        print("No dispersive materials found to plot.")
        return
 
    for mat in to_plot:
        if isinstance(mat, DispersiveBiaxialMaterial):
            axes_defs = [
                ('xx', 'C0',         'C1',         lambda l: (float(mat._na(l)), float(mat._ka(l)))),
                ('yy', 'darkmagenta','olivedrab',  lambda l: (float(mat._nb(l)), float(mat._kb(l)))),
                ('zz', 'goldenrod',  'teal',        lambda l: (float(mat._nc(l)), float(mat._kc(l)))),
            ]
        elif isinstance(mat, DispersiveMaterial):
            axes_defs = [
                ('b', 'darkmagenta', 'olivedrab',
                 lambda l: (float(mat._nb(l)), float(mat._kb(l)))),
            ]
        elif isinstance(mat, DispersiveIsotropicMaterial):
            axes_defs = [
                ('b', 'darkmagenta', 'olivedrab',
                 lambda l: (float(mat._nb(l)), float(mat._kb(l)))),
            ]
        else:
            axes_defs = [
                ('xx', 'C0', 'C1', lambda l: (
                    float(np.real(np.sqrt(mat.epsilon(l)[0][0]))),
                    float(np.imag(np.sqrt(mat.epsilon(l)[0][0]))))),
                ('yy', 'darkmagenta', 'olivedrab', lambda l: (
                    float(np.real(np.sqrt(mat.epsilon(l)[1][1]))),
                    float(np.imag(np.sqrt(mat.epsilon(l)[1][1]))))),
                ('zz', 'goldenrod', 'teal', lambda l: (
                    float(np.real(np.sqrt(mat.epsilon(l)[2][2]))),
                    float(np.imag(np.sqrt(mat.epsilon(l)[2][2]))))),
            ]
 
        n_ax = len(axes_defs)
        fig, axes = plt.subplots(1, n_ax, figsize=(5*n_ax, 3.5),
                                  sharex=True, sharey=False)
        if n_ax == 1:
            axes = [axes]
 
        for ax, (axis_label, nc, kc, nk_fn) in zip(axes, axes_defs):
            n_vals = [nk_fn(l)[0] for l in lam_plot]
            k_vals = [nk_fn(l)[1] for l in lam_plot]
            ax.plot(lam_plot, n_vals, color=nc, label=f'$n_{{{axis_label}}}$')
            ax.set_ylabel('n')
            ax.set_xlabel('$\\lambda$ (nm)')
            ax2 = ax.twinx()
            ax2.plot(lam_plot, k_vals, color=kc, label=f'$k_{{{axis_label}}}$', ls='--')
            ax2.set_ylabel('k')
            ax2.tick_params(axis='y')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(lam_min,lam_max)
 
        fig.suptitle(f'{mat.name} complex refractive index', fontsize=13)
        plt.tight_layout()
        plt.show()

def _make_unit_cell_title(layers: list, a: float) -> str:
    '''Builds compact stack title for plot_unit_cell, collapsing repeating (nH, nL) or (nL, nH) DBR pairs.'''
    # Build a flat list of (mat_name, thickness, pattern) entries, skipping Air padding
    entries = []
    for i, L in enumerate(layers):
        mat = L.material if isinstance(L.material, str) else L.get_material().name
        is_first = (i == 0)
        is_last  = (i == len(layers) - 1)
        if mat == 'Air' and L.pattern is None and (is_first or is_last):
            continue
        entries.append((mat, L.thickness, L.pattern, L.ff))

    # Greedily collapse runs of alternating 2-layer pairs (DBR detection)
    collapsed = []
    i = 0
    while i < len(entries):
        # Try to match a repeating 2-layer unit starting at i
        if i + 1 < len(entries):
            unit = (entries[i], entries[i+1])
            count = 1
            j = i + 2
            while j + 1 < len(entries) and (entries[j], entries[j+1]) == unit:
                count += 1
                j += 2
            if count >= 2:  # only collapse if it actually repeats
                m1, t1, p1, _ = unit[0]
                m2, t2, p2, _ = unit[1]
                collapsed.append(f'({m1} {t1:.0f}nm/{m2} {t2:.0f}nm)×{count}')
                i = j
                continue
        # Not a repeating pair — format single layer normally
        mat, thick, pat, ff = entries[i]
        if thick >= 1000:
            seg = f'{mat} {thick/1000:.2f}µm'
        else:
            seg = f'{mat} {thick:.0f}nm'
        if pat in ('hole', 'pillar'):
            seg += f' ({pat} ff={ff})'
        elif pat == 'cuboids':
            seg += f' (cuboids)'
        collapsed.append(seg)
        i += 1

    return f"{'  |  '.join(collapsed)}"

def _make_z_transform(layers: list, z_positions: list,
                      compress_threshold_nm: float = 500.0,
                      compressed_height_nm: float = 300.0):
    """
    Returns a function z_real -> z_plot that compresses any layer thicker
    than compress_threshold_nm down to compressed_height_nm in plot space.
    Also returns the total plot-space height.

    Layers thinner than the threshold are plotted at true scale.
    """
    segments = []   # list of (z_real_start, z_real_end, z_plot_start, z_plot_end)
    z_plot = 0.0

    for L, z0 in zip(layers, z_positions):
        z1 = z0 + L.thickness
        dz_real = L.thickness
        dz_plot = compressed_height_nm if dz_real > compress_threshold_nm else dz_real
        segments.append((z0, z1, z_plot, z_plot + dz_plot))
        z_plot += dz_plot

    z_plot_total = z_plot

    def transform(z_real):
        """Map a real z coordinate to plot z coordinate."""
        for z0r, z1r, z0p, z1p in segments:
            if z0r <= z_real <= z1r:
                # linear interpolation within segment
                frac = (z_real - z0r) / (z1r - z0r) if (z1r - z0r) > 0 else 0.0
                return z0p + frac * (z1p - z0p)
        # clamp to ends
        return 0.0 if z_real <= 0 else z_plot_total

    return transform, z_plot_total

def _make_zticks_transformed(layers, z_positions, transform, compress_threshold_nm=500.0):
    """
    Same DBR-aware tick logic as before, but returns
    (real_z_values, plot_z_values, tick_labels).
    Labels show true thickness, plot positions use compressed coords.
    """
    entries = []
    for L, z0 in zip(layers, z_positions):
        mat = L.material if isinstance(L.material, str) else L.get_material().name
        entries.append((mat, z0, z0 + L.thickness))

    z_total = entries[-1][2]
    tick_reals = {0.0, z_total}

    i = 0
    while i < len(entries):
        mat, z0, z1 = entries[i]
        if mat == 'Air' and (i == 0 or i == len(entries) - 1):
            i += 1
            continue
        if i + 1 < len(entries):
            unit = (entries[i][0], entries[i+1][0])
            count = 1
            j = i + 2
            while (j + 1 < len(entries) and
                   entries[j][0] == unit[0] and
                   entries[j+1][0] == unit[1]):
                count += 1
                j += 2
            if count >= 2:
                tick_reals.add(entries[i][1])
                tick_reals.add(entries[j-1][2])
                i = j
                continue
        tick_reals.add(z0)
        tick_reals.add(z1)
        i += 1

    tick_reals = sorted(tick_reals)
    tick_plots  = [transform(z) for z in tick_reals]
    tick_labels = [f'{z/1000:.2f}' if z >= 1000 else f'{z/1000:.2f}'
                   for z in tick_reals]

    return tick_reals, tick_plots, tick_labels

def plot_unit_cell(conf: object = None, save_fig=False, title=None, axesOn: bool = True, gridOn: bool = True):
    '''
    2D top-down view + 3D isometric view + 3D side view of the layer stack.
    Automatically reads geometry and colors from the LAYERS list.

    Parameters
    ----------
    layers   : list of Layer (defaults to global LAYERS)
    a        : float, lattice constant in nm (defaults to global A)
    save_fig : bool
    title    : str override for figure title
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    layers = conf.layers
    a = conf.lattice_const


    # Compute absolute z positions of each layer
    z_positions = []
    z = 0.0
    for layer in layers:
        z_positions.append(z)
        z += layer.thickness
    z_total = z

    fig = plt.figure(figsize=(10,5),dpi=100, layout='tight')
    # ax2d     = fig.add_subplot(131)
    ax3d     = fig.add_subplot(121, projection='3d')
    ax3dside = fig.add_subplot(122, projection='3d')
    ax3dside.set_proj_type('ortho')

    transform, z_plot_total = _make_z_transform(layers, z_positions)

    def draw_box(ax, x0, y0, z0, dx, dy, dz, fc, ec, alpha, label=None, zsort_override=None):
        x1, y1, z1 = x0+dx, y0+dy, z0+dz
        faces = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=fc,
                                edgecolor=ec, linewidth=0.8, label=label)
        # Default: sort by layer midpoint z. Lower z0 = closer to viewer (z inverted),
        # so we want those drawn last → higher _sort_zpos.
        poly._sort_zpos = zsort_override if zsort_override is not None else (z0 + dz / 2)
        ax.add_collection3d(poly)

    def draw_hole_slab(ax, z0, dz, fc, ec, w, label=None):
        cx = cy = a / 2
        hx0, hx1 = cx - w/2, cx + w/2
        hy0, hy1 = cy - w/2, cy + w/2
        zb, zt = z0, z0 + dz
        faces = []
        for z in [zb, zt]:
            faces.append([[0,0,z],[hx0,hy0,z],[hx0,hy1,z],[0,a,z]])
            faces.append([[hx1,hy0,z],[a,0,z],[a,a,z],[hx1,hy1,z]])
            faces.append([[hx0,hy0,z],[hx1,hy0,z],[a,0,z],[0,0,z]])
            faces.append([[0,a,z],[a,a,z],[hx1,hy1,z],[hx0,hy1,z]])
        ox = [0, a, a, 0]
        oy = [0, 0, a, a]
        ix = [hx0, hx0, hx1, hx1]
        iy = [hy0, hy1, hy1, hy0]
        for k in range(4):
            k1 = (k+1) % 4
            faces.append([[ox[k],oy[k],zb],[ox[k1],oy[k1],zb],
                        [ox[k1],oy[k1],zt],[ox[k],oy[k],zt]])
            faces.append([[ix[k1],iy[k1],zb],[ix[k],iy[k],zb],
                        [ix[k],iy[k],zt],[ix[k1],iy[k1],zt]])
        poly = Poly3DCollection(faces, alpha=0.6, facecolor=fc,
                                edgecolor='none', linewidth=0, label=label)
        # Same convention: deeper layers (larger z0) drawn first
        poly._sort_zpos = (z0 + dz / 2)
        ax.add_collection3d(poly)
        ek = dict(color=ec, lw=0.8, alpha=0.7, zorder=int(-z0 * 100))
        for (x,y) in [(0,0),(a,0),(a,a),(0,a)]:
            ax.plot([x,x],[y,y],[zb,zt], **ek)
        for (x,y) in [(hx0,hy0),(hx1,hy0),(hx1,hy1),(hx0,hy1)]:
            ax.plot([x,x],[y,y],[zb,zt], **ek)
        for z in [zb, zt]:
            ax.plot([0,a,a,0,0],[0,0,a,a,0],[z]*5, **ek)
            ax.plot([hx0,hx1,hx1,hx0,hx0],[hy0,hy0,hy1,hy1,hy0],[z]*5, **ek)
    
    # -- If layers have drastically different thicknesses, may want to compress some in z:
    def draw_box_t(ax, x0, y0, z0_real, dx, dy, dz_real, fc, ec, alpha, label=None, zsort_override=None):
            """draw_box but with z coordinates transformed to plot space."""
            z0p = transform(z0_real)
            z1p = transform(z0_real + dz_real)
            draw_box(ax, x0, y0, z0p, dx, dy, z1p - z0p, fc, ec, alpha, label, zsort_override)

    def draw_hole_slab_t(ax, z0_real, dz_real, fc, ec, w, label=None):
        z0p = transform(z0_real)
        z1p = transform(z0_real + dz_real)
        draw_hole_slab(ax, z0p, z1p - z0p, fc, ec, w, label)

    # -- Draw each layer in 3D ────────────────────────────────────────────────
    legend_drawn = set()

    for layer, z0 in zip(layers, z_positions):
        mat_name = (layer.material if isinstance(layer.material, str)
                    else layer.get_material().name)
        fc, ec, al = _mat_color(mat_name)
        h  = layer.thickness
        cx = cy = a / 2

        z0p = transform(z0)
        z1p = transform(z0 + h)

        lbl = mat_name if mat_name not in legend_drawn else None
        legend_drawn.add(mat_name)

        if layer.pattern is None:
            for ax in [ax3d, ax3dside]:
                draw_box_t(ax, 0, 0, z0, a, a, h, fc, ec, al, label=lbl)
                lbl = None

        elif layer.pattern == 'pillar':
            w = layer.ff * a
            air_fc, air_ec, air_al = _mat_color('Air')
            for ax in [ax3d, ax3dside]:
                draw_box_t(ax, 0, 0, z0, a, a, h, air_fc, air_ec, air_al, label=None)
                draw_box_t(ax, cx-w/2, cy-w/2, z0, w, w, h, fc, ec, 0.9, label=lbl)
                lbl = None

        elif layer.pattern == 'hole':
            w = layer.ff * a
            for ax in [ax3d, ax3dside]:
                draw_hole_slab_t(ax, z0, h, fc, ec, w,label=lbl)
                lbl = None

        elif layer.pattern == 'cuboids':
            w1, w2, x1, y1, x2, y2 = layer.get_cuboid_geometry(a)
            air_fc, air_ec, air_al = _mat_color('Air')
            for ax in [ax3d, ax3dside]:
                draw_box_t(ax, 0, 0, z0, a, a, h, air_fc, air_ec, air_al, label=None)
                draw_box_t(ax, x1-w1/2, y1-w1/2, z0, w1, w1, h, fc, ec, 0.9, label=lbl)
                draw_box_t(ax, x2-w2/2, y2-w2/2, z0, w2, w2, h, fc, ec, 0.9, label=None)
                lbl = None

        else:
            import warnings
            warnings.warn(f"Unrecognized pattern '{layer.pattern}' for layer '{mat_name}' — layer will not be drawn.")

    # -- 3D axis formatting ────────────────────────────────────────────────────

    ztick_reals, ztick_plots, ztick_labels = _make_zticks_transformed(
        layers, z_positions, transform)

    max_label_len = max(len(s) for s in ztick_labels)
    tick_pad      = max_label_len * 0
    zlabel_pad    = tick_pad + 35

    for ax, view_angles, title_str in [
        (ax3d,     (20, -50), '3D view'),
        (ax3dside, (0,   90), 'Side view'),
    ]:
        match title_str:
            case '3D view':
                ax.set_xlim(0, a)
                ax.set_ylim(0, a)
                n_ticks = 5  # 0, two middle, max
                raw = np.linspace(0, a, n_ticks)
                ticks = [0] + [round(v / 10) * 10 for v in raw[1:-1]] + [int(a)]
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

                ax.set_zlim(0, z_plot_total)
                ax.set_xlabel('x (nm)', fontsize=10)
                ax.set_ylabel('y (nm)', fontsize=10)
                
                ax.zaxis.set_rotate_label(False)
                ax.set_zlabel(f'z\n$\downarrow$', fontsize=10, labelpad=-10)
                # ax.set_zticks(ztick_plots)
                ax.set_zticks([])
                # ax.set_zticklabels(ztick_labels, fontsize=6)
                ax.invert_zaxis()
                # ax.tick_params(labelsize=10)
                ax.set_title(title_str, fontsize=11,fontweight='bold',y=1.15)
                ax.view_init(*view_angles)
                leg=ax.legend(fontsize=10, loc='upper left',framealpha=1)
                leg.set_zorder(2000)
                ax.set_box_aspect((1,1,1.5))
                handles, labels = ax.get_legend_handles_labels()
            case 'Side view':
                ax.set_xlim(0, a)
                ax.set_ylim(0, a)
                ax.set_zlim(0, z_plot_total)
                ax.set_xlabel('x (nm)', fontsize=10,labelpad=0)
                n_ticks = 5  # 0, two middle, max
                raw = np.linspace(0, a, n_ticks)
                ticks = [0] + [round(v / 10) * 10 for v in raw[1:-1]] + [int(a)]
                ax.set_xticks(ticks)

                # ax.set_ylabel('y (nm)', fontsize=10)
                ax.invert_xaxis()
                ax.invert_zaxis()
                ax.set_yticks([])
                ax.zaxis.set_rotate_label(False)
                ax.set_zlabel('z (µm)\n$\downarrow$', fontsize=10,labelpad=zlabel_pad)
                ax.set_zticks(ztick_plots)
                ax.set_zticklabels(ztick_labels, fontsize=10, ha='right')
                ax.tick_params(axis='z',pad=tick_pad)
                # ax.tick_params(labelsize=10)
                ax.set_title(title_str, fontsize=11,y=-0.05,fontweight='bold')
                ax.view_init(*view_angles)
                # ax.legend(fontsize=10, loc='upper left')
                ax.set_box_aspect((1.25,1,2))
                # ax.xaxis._axinfo['juggled']=(0,1,1)
                leg2 = ax.legend(handles,labels,loc=(0.7,0.2),framealpha=1)
                leg2.set_zorder(2000)

    # -- Figure title ──────────────────────────────────────────────────────────
    if title is None:
        title = _make_unit_cell_title(layers, a)
    fig.suptitle(title, fontsize=10, ha='center',va='bottom',wrap=True)
    # fig.subplots_adjust(bottom=0.15)
    fig.tight_layout()

    if save_fig:
        if not conf:
            print(f"If you'd like files saved, please provide valid conf = ru.RCWAConfig.")
        else:
            fname = conf.output_dir / f"unitcell_{make_stack_slug(conf)}.pdf"
            if fname.exists():
                stem = fname.stem
                v = 2
                while True:
                    candidate = Path(conf.output_dir) / f'{stem}_v{v}.pdf'
                    if not candidate.exists():
                        fname = candidate
                        break
                    v += 1
            fig.savefig(fname, format='pdf', bbox_inches='tight')
            print(f"Saved {fname}")
    return fig
 
 # -- Core S4 functions

def _resolve_rotation(layer, conf):
    '''
    Resolves effective rotation/tilt angles for a layer.
    Priority: layer (if set, including 0) --> conf.global_ --> 0 (fallback)
    Returns (rot_deg, tilt_deg, tilt_azim_deg) as floats -- never None.
    '''
    match layer.layer_rot:
        case None: rot = conf.global_rot if conf.global_rot is not None else 0.0
        case _:    rot = layer.layer_rot
    return float(rot)

def build_simulation(conf: object, 
                    lam_nm: float,
                    # NOTE: Only above parameters necessary to run, if parameters below
                    # are passed they will override config setting. This may affect json
                    # saving, so be careful. Running outside of global config settings
                    # should only be usd for running tests, not for publication grade/reproducible
                    # simulations. 
                    a: float = None,
                    layers: list = None,
                    n_basis: int = None,
                    n_jobs: int = None) -> 'S4.Simulation':

    """
    Build an S4 simulation from an ordered list of Layer objects.

    Parameters
    ----------
    conf      : global config settings
    lam_nm    : wavelength in nm — sets dispersive material values

    layers    : list of Layer, ordered top (incident) to bottom (substrate)
    n_basis   : number of Fourier basis functions
    n_jobs    : number of simultaneous jobs
    """
    a = a or conf.lattice_const
    layers = layers or conf.layers
    n_basis = n_basis or conf.n_basis
    n_jobs = n_jobs or conf.n_jobs

    S = S4.New(Lattice=((a, 0), (0, a)), NumBasis=n_basis)

    # Always register Air — needed as background for patterned layers
    S.SetMaterial(Name='Air', Epsilon=((1,0,0),(0,1,0),(0,0,1)))

    # Register all unique materials
    # Key = (material_name, effective_phiz, theta_tilt, phi_az) so that the same
    # material with different orientations gets separate S4 material names
    s4_name_map = {}   # key -> s4_name string

    for i, layer in enumerate(layers):
        mat = layer.get_material()

        # Effective rot: use layer-level value if explicitly set (INCLUDING SET TO 0),
        # otherwise apply the global_rot sweep value
        # same for tilt and tilt_azim
        match layer.layer_rot:
            case None:
                match conf.global_rot:
                    case 0.0 | None:
                        eff_rot = 0.0
                    case _:
                        eff_rot = conf.global_rot
            case _:
                eff_rot = layer.layer_rot
  

        key = (mat.name, eff_rot)

        if key not in s4_name_map:
            # Generate a unique S4 material name
            s4_name = f"{mat.name}_{len(s4_name_map)}"
            s4_name_map[key] = s4_name

            # Compute epsilon at this wavelength and orientation
            eps = mat.epsilon(lam_nm,
                              rot       = eff_rot,
                            )
            S.SetMaterial(Name=s4_name, Epsilon=eps)

        # Stash the S4 name on the layer for AddLayer below
        layer._s4_name = s4_name_map[key]

    # Add layers and patterns
    for i, layer in enumerate(layers):
        layer_name = f'L{i}'
        s4_mat     = layer._s4_name

        if layer.pattern is None:
            S.AddLayer(Name=layer_name,
                       Thickness=layer.thickness,
                       Material=s4_mat)

        elif layer.pattern == 'pillar':
            w = layer.ff * a
            S.AddLayer(Name=layer_name,
                       Thickness=layer.thickness,
                       Material='Air')
            S.SetRegionRectangle(Layer=layer_name, Material=s4_mat,
                                  Center=(a/2, a/2),
                                  Halfwidths=(w/2, w/2), Angle=0)

        elif layer.pattern == 'hole':
            w = layer.ff * a
            S.AddLayer(Name=layer_name,
                       Thickness=layer.thickness,
                       Material=s4_mat)
            S.SetRegionRectangle(Layer=layer_name, Material='Air',
                                  Center=(a/2, a/2),
                                  Halfwidths=(w/2, w/2), Angle=0)

        elif layer.pattern == 'cuboids':
            w1, w2, x1, y1, x2, y2 = layer.get_cuboid_geometry(a)
            S.AddLayer(Name=layer_name,
                       Thickness=layer.thickness,
                       Material='Air')
            S.SetRegionRectangle(Layer=layer_name, Material=s4_mat,
                                  Center=(x1, y1),
                                  Halfwidths=(w1/2, w1/2), Angle=0)
            S.SetRegionRectangle(Layer=layer_name, Material=s4_mat,
                                  Center=(x2, y2),
                                  Halfwidths=(w2/2, w2/2), Angle=0)
        else:
            raise ValueError(f"Unknown pattern '{layer.pattern}' in layer {i}")

    return S

def update_simulation_materials(S, 
                                lam_nm: float,
                                conf: object = None,
                                layers: list = None
                                ):
    """
    Update material permittivities on an S4 simulation object
    for a new wavelength, without rebuilding geometry.
    Called in the inner wavelength loop of workers.
    Either pass an explicit list of layers or a config file, which will override with its
    own conf.layers.
    """
    if not (layers or conf):
        raise Exception("To update a simulation, please provide either a config object or a list of layers.")
    match conf:
        case None:
            layers = layers
        case _:
            layers = conf.layers
    updated = set()
    for layer in layers:
        mat      = layer.get_material()
        eff_rot = _resolve_rotation(layer, conf)
        key      = (mat.name, eff_rot)

        if key not in updated and hasattr(layer, '_s4_name'):
            eps = mat.epsilon(lam_nm,
                            eff_rot,
                            )
            S.SetMaterial(Name=layer._s4_name, Epsilon=eps)
            updated.add(key)

# ── Stack geometry helpers, Jones matrix sampling ──────────────────────────────────

# def sp_basis(elev_deg, azim_deg):   # NOTE: not using this anymore, S4 has native polarization basis
#     # Convert elev_deg into what S4 documentation specifies
#     # s4phi = 90-elev_deg; s4theta = azim_deg
#     th = np.radians(elev_deg); ph = np.radians(azim_deg)
#     khat = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
#     if np.abs(np.sin(th)) < 1e-8:
#         return np.array([0.,1.,0.]), np.array([1.,0.,0.])
#     zhat = np.array([0.,0.,1.])
#     s = np.cross(khat, zhat); s /= np.linalg.norm(s)
#     p = np.cross(s, khat)
#     return s, p

# def get_jones_matrices(S_sim, elev_deg, azim_deg, conf):
#     phi   = abs(float(elev_deg))
#     theta = float(azim_deg) if elev_deg >= 0 else (float(azim_deg) + 180) % 360

#     n_G         = len(S_sim.GetBasisSet())
#     first_layer = 'L0'
#     last_layer  = f'L{len(conf.layers)-1}'

#     # Detect S4's s/p slot ordering at this angle.
#     # Send in pure s, see which amplitude slot carries the energy.
#     # Whichever of back[0] or back[n_G] is larger is the s-slot.
#     S_sim.SetExcitationPlanewave(
#         IncidenceAngles=(phi, theta),
#         sAmplitude=1, pAmplitude=0, Order=0)
#     _, back_probe = S_sim.GetAmplitudes(Layer=first_layer, zOffset=0)
#     # For a high-reflectance structure the reflected s amplitude dominates;
#     # for low-R structures use transmission instead as the probe
#     forw_probe, _ = S_sim.GetAmplitudes(Layer=last_layer, zOffset=0)
    
#     # s-slot is whichever has larger total amplitude (reflected + transmitted)
#     amp_slot0  = abs(back_probe[0])   + abs(forw_probe[0])
#     amp_slot1  = abs(back_probe[n_G]) + abs(forw_probe[n_G])
#     s_slot = 0 if amp_slot0 >= amp_slot1 else n_G
#     p_slot = n_G if s_slot == 0 else 0

#     T = np.zeros((2, 2), dtype=complex)
#     R = np.zeros((2, 2), dtype=complex)

#     for j, (sa, pa) in enumerate([(1., 0.), (0., 1.)]):
#         S_sim.SetExcitationPlanewave(
#             IncidenceAngles=(phi, theta),
#             sAmplitude=sa, pAmplitude=pa, Order=0)

#         _, back = S_sim.GetAmplitudes(Layer=first_layer, zOffset=0)
#         R[0, j] = back[s_slot]
#         R[1, j] = back[p_slot]

#         forw, _ = S_sim.GetAmplitudes(Layer=last_layer, zOffset=0)
#         T[0, j] = forw[s_slot]
#         T[1, j] = forw[p_slot]

#     return T, R

# def get_jones_matrices(S_sim, elev_deg, azim_deg, conf):
#     phi   = abs(float(elev_deg))
#     theta = float(azim_deg) if elev_deg >= 0 else (float(azim_deg) + 180) % 360

#     n_G         = len(S_sim.GetBasisSet())
#     first_layer = 'L0'
#     last_layer  = f'L{len(conf.layers)-1}'

#     # S4 slot 0 corresponds to E along y (s at azim=0)
#     # S4 slot n_G corresponds to E along x (s at azim=90)
#     # For arbitrary azimuth, the true s direction is perpendicular
#     # to the plane of incidence: s_hat = (-sin(azim), cos(azim))
#     # in the xy plane.
#     # S4 internal basis: slot 0 ~ y-component, slot n_G ~ x-component
#     # So: s_amplitude = -sin(azim)*slot[n_G] + cos(azim)*slot[0]
#     #     p_amplitude =  cos(phi)*(cos(azim)*slot[n_G] + sin(azim)*slot[0])
#     # (the cos(phi) factor accounts for p having an in-plane component
#     # reduced by the polar angle, but for amplitude extraction from
#     # the 0th order we just need the in-plane projection)

#     az = np.radians(float(azim_deg))
#     cs, ss = np.cos(az), np.sin(az)

#     def extract_sp(amps):
#         a0  = amps[0]       # y-component (slot 0)
#         a1  = amps[n_G]     # x-component (slot n_G)
#         s_amp =  cs * a0 - ss * a1   # E perpendicular to plane of incidence
#         p_amp =  ss * a0 + cs * a1   # E parallel to plane of incidence
#         return s_amp, p_amp

#     T = np.zeros((2, 2), dtype=complex)
#     R = np.zeros((2, 2), dtype=complex)

#     for j, (sa, pa) in enumerate([(1., 0.), (0., 1.)]):
#         S_sim.SetExcitationPlanewave(
#             IncidenceAngles=(phi, theta),
#             sAmplitude=sa, pAmplitude=pa, Order=0)

#         _, back = S_sim.GetAmplitudes(Layer=first_layer, zOffset=0)
#         forw, _ = S_sim.GetAmplitudes(Layer=last_layer,  zOffset=0)

#         R[0, j], R[1, j] = extract_sp(back)
#         T[0, j], T[1, j] = extract_sp(forw)

#     return T, R

def get_jones_matrices(S_sim, elev_deg, azim_deg, conf):
    phi   = abs(float(elev_deg))
    theta = float(azim_deg) if elev_deg >= 0 else (float(azim_deg) + 180) % 360

    n_G         = len(S_sim.GetBasisSet())
    first_layer = 'L0'
    last_layer  = f'L{len(conf.layers)-1}'

    az = np.radians(float(azim_deg))
    cs, ss = np.cos(az), np.sin(az)

    def extract_sp(amps):
        a0 = amps[0]
        a1 = amps[n_G]
        return cs*a0 + ss*a1, -ss*a0 + cs*a1

    T = np.zeros((2, 2), dtype=complex)
    R = np.zeros((2, 2), dtype=complex)

    for j, (sa, pa) in enumerate([(1., 0.), (0., 1.)]):
        S_sim.SetExcitationPlanewave(
            IncidenceAngles=(phi, theta),
            sAmplitude=sa, pAmplitude=pa, Order=0)

        _, back = S_sim.GetAmplitudes(Layer=first_layer, zOffset=0)
        forw, _ = S_sim.GetAmplitudes(Layer=last_layer,  zOffset=0)

        R[0, j], R[1, j] = extract_sp(back)
        T[0, j], T[1, j] = extract_sp(forw)

    return T, R

# ── Post-processing ────────────────────────────────────────────────────────
def jones_to_circular(J: np.ndarray) -> np.ndarray:
    """Convert 2×2 Jones matrix from linear (s,p) to circular (R,L) basis."""
    U = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    return np.conj(U).T @ J @ U

def compute_observables(T: np.ndarray, R: np.ndarray) -> dict:
    """
    Store only raw Jones amplitudes. All derived quantities are computed
    at plot time by _jones_cols_to_obs().
 
    Columns saved:
        T_j00_re, T_j00_im, T_j10_re, T_j10_im,
        T_j01_re, T_j01_im, T_j11_re, T_j11_im,
        R_j00_re, R_j00_im, ... (same for R)
    """
    def _components(J, prefix):
        return {
            f'{prefix}j00_re': float(J[0, 0].real),
            f'{prefix}j00_im': float(J[0, 0].imag),
            f'{prefix}j10_re': float(J[1, 0].real),
            f'{prefix}j10_im': float(J[1, 0].imag),
            f'{prefix}j01_re': float(J[0, 1].real),
            f'{prefix}j01_im': float(J[0, 1].imag),
            f'{prefix}j11_re': float(J[1, 1].real),
            f'{prefix}j11_im': float(J[1, 1].imag),
        }
    return {**_components(T, 'T_'), **_components(R, 'R_')}

def _jones_cols_to_obs(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Reconstruct all derived observables from stored Jones columns.
    Operates fully vectorized — no row-wise loops.
 
    prefix : 'T_' or 'R_'
 
    Returns a DataFrame with columns named without prefix, e.g.:
        ss, ps, sp, pp, S0s, S1s, S2s, S3s, S0p, S1p, S2p, S3p,
        RR, LL, RL, LR, CD, total
    Call as:  obs = _jones_cols_to_obs(df, 'R_')
    Then:     df['R_ss'] = obs['ss']
    """
    p = prefix
    j00 = df[f'{p}j00_re'].values + 1j * df[f'{p}j00_im'].values
    j10 = df[f'{p}j10_re'].values + 1j * df[f'{p}j10_im'].values
    j01 = df[f'{p}j01_re'].values + 1j * df[f'{p}j01_im'].values
    j11 = df[f'{p}j11_re'].values + 1j * df[f'{p}j11_im'].values
 
    # Stack into (N, 2, 2) for batched circular transform
    J = np.stack([np.stack([j00, j01], axis=-1),
                  np.stack([j10, j11], axis=-1)], axis=-2)  # (N, 2, 2)
    U  = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    Uh = np.conj(U).T
    Jc = Uh[None] @ J @ U[None]  # (N, 2, 2)
    RR = Jc[:, 0, 0]; RL = Jc[:, 0, 1]
    LR = Jc[:, 1, 0]; LL = Jc[:, 1, 1]

    # Unpolarized input Stokes
    S0u = (np.abs(j00)**2 + np.abs(j10)**2 +
        np.abs(j01)**2 + np.abs(j11)**2) / 2
    S1u = ((np.abs(j00)**2 - np.abs(j10)**2) +
        (np.abs(j01)**2 - np.abs(j11)**2)) / 2
    S2u = (np.real(j00*np.conj(j10)) +
        np.real(j01*np.conj(j11)))
    S3u = (np.imag(j00*np.conj(j10)) +
        np.imag(j01*np.conj(j11)))
    
    # Mask psi and chi where output power is too low to be meaningful
    mask_s = np.abs(j00)**2 + np.abs(j10)**2 < 1e-3   # s-input too weak
    mask_p = np.abs(j01)**2 + np.abs(j11)**2 < 1e-3   # p-input too weak
    mask_u = S0u < 1e-3                                 # unpolarized too weak

    psi_s_vals = 0.5 * np.degrees(np.arctan2(
                    2*np.real(j00*np.conj(j10)),
                    np.abs(j00)**2 - np.abs(j10)**2))
    chi_s_vals =  0.5 * np.degrees(np.arcsin(np.clip(
                2*np.imag(j00*np.conj(j10)) /
                (np.abs(j00)**2 + np.abs(j10)**2 + 1e-30), -1, 1)))
    psi_p_vals =  0.5 * np.degrees(np.arctan2(
                    2*np.real(j01*np.conj(j11)),
                    np.abs(j01)**2 - np.abs(j11)**2))
    chi_p_vals =  0.5 * np.degrees(np.arcsin(np.clip(
                    2*np.imag(j01*np.conj(j11)) /
                    (np.abs(j01)**2 + np.abs(j11)**2 + 1e-30), -1, 1)))
    psi_u_vals = 0.5 * np.degrees(np.arctan2(S2u, S1u))
    chi_u_vals =  0.5 * np.degrees(np.arcsin(np.clip(
                    S3u / (S0u + 1e-30), -1, 1)))

    psi_s_vals[mask_s] = np.nan
    psi_p_vals[mask_p] = np.nan
    psi_u_vals[mask_u] = np.nan
    chi_s_vals[mask_s] = np.nan
    chi_p_vals[mask_p] = np.nan
    chi_u_vals[mask_u] = np.nan
 
    out = pd.DataFrame({
        # Intensity Jones elements
        'ss':    np.abs(j00)**2,
        'ps':    np.abs(j10)**2,
        'sp':    np.abs(j01)**2,
        'pp':    np.abs(j11)**2,
        # Stokes for s-input
        'S0s':   np.abs(j00)**2 + np.abs(j10)**2,
        'S1s':   np.abs(j00)**2 - np.abs(j10)**2,
        'S2s':   2 * np.real(j00 * np.conj(j10)),
        'S3s':   2 * np.imag(j00 * np.conj(j10)),
        # Stokes for p-input
        'S0p':   np.abs(j11)**2 + np.abs(j01)**2,
        'S1p':   np.abs(j11)**2 - np.abs(j01)**2,
        'S2p':   2 * np.real(j11 * np.conj(j01)),
        'S3p':   2 * np.imag(j11 * np.conj(j01)),
        # Circular basis
        'RR':    np.abs(RR)**2,
        'LL':    np.abs(LL)**2,
        'RL':    np.abs(RL)**2,
        'LR':    np.abs(LR)**2,
        'CD':    np.abs(RR)**2 - np.abs(LL)**2,
        # Polarization angle and ellipticity for s-input
        'psi_s':  psi_s_vals,
        'chi_s':  chi_s_vals,
        # For p-input
        'psi_p':  psi_p_vals,  
        'chi_p':  chi_p_vals,
        # For unpolarized input
        'psi_u':  psi_u_vals,
        'chi_u':  chi_u_vals,
        # Total Stokes parameters for "unpolarized" input
        'S0u':  S0u,                        # = total output
        'S1u':  S1u,                        # linear TE-TM for unpolarized input
        'S2u':  S2u,                        # diagonal linear for unpolarized input
        'S3u':  S3u,                        # circular for unpolarized input
        'S1un': S1u / (S0u + 1e-30),        # normalized versions
        'S2un': S2u / (S0u + 1e-30),
        'S3un': S3u / (S0u + 1e-30),
        # For unpolarized input, the degree of circular polarization of the output:
        'DCP_u': S3u / (S0u + 1e-30),
    }, index=df.index)
    return out

def _add_obs(df: pd.DataFrame) -> pd.DataFrame:
    for prefix in ('T_', 'R_'):
        obs = _jones_cols_to_obs(df, prefix)
        for col in obs.columns:
            full_col = f'{prefix}{col}'
            if full_col not in df.columns:
                df[full_col] = obs[col].values
    return df

# - Convergence tests
def convergence_test(conf: object, lam_nm: float = None, elev_deg=5.0, azim_deg=0.0,
                     basis_vals=(25, 49, 81, 100, 144, 196, 256)):
    """
    Sweep NumBasis and report T_ss, T_pp convergence at a single wavelength.
    Uses the median wavelength by default — or pass lam_nm explicitly to test
    at a resonance where convergence is hardest.
    """
    layers = conf.layers
    lam_nm = lam_nm or float(np.median(conf.wavelengths))
    rot = conf.global_rot if conf.global_rot is not None else 0.0

    # print()
    print(f'%--'*30)
    print(f"Convergence test: λ={lam_nm:.0f} nm  elev={elev_deg:.1f}°  "
          f"azim={azim_deg:.0f}°  rot={rot:.0f}°")
    print(f"Stack: {' | '.join(repr(L) for L in layers)}")
    print(f"| {'NumBasis':>10}  {'T_ss':>10}  {'T_pp':>10}  {'|dT_ss|':>10}  {'time(ms)':>10}")

    prev = None
    for nb in basis_vals:
        t0 = time.perf_counter()
        S  = build_simulation(conf=conf, lam_nm=lam_nm, n_basis=nb)
        S.SetFrequency(1.0 / lam_nm)
        T, _ = get_jones_matrices(S, elev_deg, azim_deg, conf)
        dt   = (time.perf_counter() - t0) * 1e3
        Tss  = abs(T[0,0])**2
        Tpp  = abs(T[1,1])**2
        delta = abs(Tss - prev) if prev is not None else float('nan')
        print(f"| {nb:>10d}  {Tss:>10.5f}  {Tpp:>10.5f}  {delta:>10.5f}  {dt:>10.1f}")
        prev = Tss
    # print(f'%'*90)

def convergence_test_phase(conf:object, lam_nm=None, elev_deg=5.0, azim_deg=0.0,
                            basis_vals=(49, 81, 100, 144, 196, 256)):
    """
    Sweep NumBasis and report phase of t_ss and CD — these converge faster
    than amplitude and are the quantities that matter for polarization topology.
    """

    if lam_nm is None:
        lam_nm = float(np.median(conf.wavelengths))
    rot = conf.global_rot if conf.global_rot is not None else 0.0

    # print()
    print(f'%--'*30)
    print(f"Phase convergence: λ={lam_nm:.0f} nm  elev={elev_deg:.1f}°  "
          f"azim={azim_deg:.0f}°    rot={rot:.0f}°  ")
    print(f"| {'NumBasis':>10}  {'|t_ss|':>10}  {'phase_ss (deg)':>16}  "
          f"{'CD':>10}  {'time(ms)':>10}")
    for nb in basis_vals:
        t0   = time.perf_counter()
        S    = build_simulation(conf=conf, lam_nm=lam_nm)
        S.SetFrequency(1.0 / lam_nm)
        T, _ = get_jones_matrices(S, elev_deg, azim_deg, conf=conf)
        dt   = (time.perf_counter() - t0) * 1e3
        phase = np.degrees(np.angle(T[0,0]))
        Tc    = jones_to_circular(T)
        CD    = abs(Tc[0,0])**2 - abs(Tc[1,1])**2
        print(f"| {nb:>10d}  {abs(T[0,0]):>10.5f}  {phase:>16.3f}  "
              f"{CD:>10.5f}  {dt:>10.1f}")
    # print(f'%'*90)

# ── Wavelength scan (to find resonances) ─────────────────────────────────────────

def scan_wavelengths(conf: object, lam_vals = None, n_basis=25, elev_deg=0.0, azim_deg=0.0):
    """
    Scan T_ss and T_pp across wavelengths at normal (or near-normal) incidence
    to locate resonance dips before running full sweeps.
    """
  
    lam_vals = lam_vals or conf.wavelengths[::4]  # every 5th point for speed
    n_basis = n_basis or conf.n_basis
    rot = conf.global_rot if conf.global_rot is not None else 0.0

    # print()
    print(f'%--'*30)
    print(f"Wavelength scan: elev={elev_deg:.1f}°  azim={azim_deg:.0f}°  "
          f"rot={rot:.0f}°      N_BASIS={n_basis}")
    print(f"{'lam (nm)':>10}  {'energy (eV)':>12}  {'T_ss':>10}  "
          f"{'T_pp':>10}  {'T_ss+T_pp':>12}")

    # Build once, update frequency each step (geometry doesn't change)
    S = build_simulation(conf, float(lam_vals[0]))

    for lam in lam_vals:
        update_simulation_materials(S, conf=conf, lam_nm=float(lam))
        S.SetFrequency(1.0 / lam)
        T, _ = get_jones_matrices(S, elev_deg, azim_deg, conf=conf)
        Tss  = abs(T[0,0])**2
        Tpp  = abs(T[1,1])**2
        print(f"{lam:>10.1f}  {1239.8/lam:>12.4f}  {Tss:>10.5f}  "
              f"{Tpp:>10.5f}  {Tss+Tpp:>12.5f}")
    # print(f'%'*90)

# ── Energy conservation check ─────────────────────────────────────────────────

def verify_energy_conservation(conf: object, lam_nm=None, elev_deg=10.0,
                                azim_deg=0.0, rot_deg=0.0):
    """
    Compute Jones T matrix and cross-check with GetPowerFlux.
    T + R + A = 1 for lossless materials; A > 0 indicates absorption or
    numerical error from oblique incidence (see notes in appendix).
    Uses separate fresh S4 objects for Jones and PowerFlux to avoid
    state contamination between solves.
    """
    layers = conf.layers
    lam_nm = lam_nm or float(np.median(conf.wavelengths))
    rot = rot_deg or conf.global_rot or 0.0

    first_layer  = layers[0].get_material().name
    last_layer   = layers[-1].get_material().name

    # print()
    print(f'%--'*30)
    print(f"Energy conservation check:")
    print(f"|  λ={lam_nm:.0f} nm  elev={elev_deg:.1f}°  "
          f"azim={azim_deg:.0f}°  rot={rot:.0f}°    ")
    print(f"|  Stack: {' | '.join(repr(L) if isinstance(L.material,str) else L.get_material().name for L in layers)}")
    print(f"|  Incident medium: {first_layer}  |  Substrate: {last_layer}")

    # ── Jones matrix (fresh simulation) ───────────────────────────────────────
    S = build_simulation(conf, lam_nm)
    S.SetFrequency(1.0 / lam_nm)
    T, _ = get_jones_matrices(S, elev_deg, azim_deg, conf)

    print(f"Jones T matrix:")
    print(f"|           s-in           p-in")
    print(f"|  s-out  {T[0,0]:>+.4f}   {T[0,1]:>+.4f}")
    print(f"|  p-out  {T[1,0]:>+.4f}   {T[1,1]:>+.4f}")
    print(f"V")

    # ── PowerFlux check (separate fresh simulation per polarization) ───────────
    # GetPowerFlux Layer names are the layer names assigned in build_simulation:
    # L0 = first layer (incident), L{n-1} = last layer (substrate/transmission)
    first_layer_name = 'L0'
    last_layer_name  = f'L{len(layers)-1}'

    print(f"{'input':>6}  {'T_jones':>10}  {'T_pf':>10}  "
          f"{'R_pf':>10}  {'A':>10}  {'T+R+A':>10}")
    print('-'*68)
    for j, (sa, pa, label) in enumerate([(1,0,'s-in'), (0,1,'p-in')]):
        S2 = build_simulation(conf, lam_nm)
        S2.SetFrequency(1.0 / lam_nm)
        S2.SetExcitationPlanewave(IncidenceAngles=(elev_deg, azim_deg),
                                   sAmplitude=sa, pAmplitude=pa, Order=0)

        fwd_t, _       = S2.GetPowerFlux(Layer=last_layer_name,  zOffset=50.0)
        fwd_r, bwd_r   = S2.GetPowerFlux(Layer=first_layer_name, zOffset=50.0)

        Tpow   = fwd_t.real
        Rpow   = -bwd_r.real
        Apow   = 1 - Tpow - Rpow
        Tjones = abs(T[0,j])**2 + abs(T[1,j])**2

        print(f"{label:>6}  {Tjones:>10.5f}  {Tpow:>10.5f}  "
              f"{Rpow:>10.5f}  {Apow:>10.5f}  {Tpow+Rpow+Apow:>10.6f}")


# -------- STUDY DEFINITIONS --------------------
# —— Study/config loader
def load_study(path) -> tuple[pd.DataFrame, object]:
    '''
    Load a matched CSV + JSON pair by base path (with or without extension).
    Returns (df, conf) — either may be None if the file wasn't found.

    Example:
        df, conf = ru.load_study('/path/to/study1_A500nm_..._nb100')
    '''
    base = str(path).removesuffix('.csv').removesuffix('.json')
    csv_path  = Path(base + '.csv')
    json_path = Path(base + '.json')

    df   = None
    conf = None

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print(f"Warning: no CSV found at {csv_path.name}. No data loaded.")

    if json_path.exists():
        conf = RCWAConfig.load(json_path)
    else:
        print(f"Warning: no JSON found at {json_path.name}. No config loaded.")

    if df is not None and conf is None:
        print("\nData loaded but config not found — plotting functions that require conf will not work.\n")
    if conf is not None and df is None:
        print("\nConfig loaded but data not found. You can re-run studies with this config.\n")
    if df is not None and conf is not None:
        print(f"\nLoaded {len(df)} rows  |  {csv_path.name}\n")

    return df, conf

# ── Stack metadata helpers ────────────────────────────────────────────────────

def stack_label(layers: list, a: float) -> str:
    """
    Build a concise one-line label summarising the layer stack and period.
    Example: 'P=500 nm | Air 500 | CrSBr 70 (hole ff=0.5) | SiO2 100 | Air 500'
    """
    parts = []
    for L in layers:
        mat = L.material if isinstance(L.material, str) else L.get_material().name
        pat = ''
        if L.pattern in ('hole', 'pillar'):
            pat = f' ({L.pattern} ff={L.ff})'
        elif L.pattern == 'cuboids':
            pat = f' (cuboids w0={L.w0} a={L.alpha})'
        parts.append(f'{mat} {L.thickness:.0f}nm{pat}')
    return f'P={a:.0f} nm  |  ' + '  |  '.join(parts)

def stack_title(layers: list, a: float, extras: str = '') -> str:
    """Multi-line figure title with stack summary and optional extra info."""
    lines = [stack_label(layers, a)]
    if extras:
        lines.append(extras)
    return '\n'.join(lines)

def _get_patterned_layer_info(layers: list) -> str:
    """Return a short string describing patterned layers for plot subtitles."""
    infos = []
    for L in layers:
        if L.pattern is None:
            continue
        mat = L.material if isinstance(L.material, str) else L.get_material().name
        if L.pattern in ('hole', 'pillar'):
            infos.append(f'{mat} {L.pattern} ff={L.ff} h={L.thickness:.0f}nm')
        elif L.pattern == 'cuboids':
            infos.append(f'{mat} cuboids w0={L.w0} a={L.alpha} h={L.thickness:.0f}nm')
    return '  |  '.join(infos) if infos else 'unpatterned'

class ProgressCallback:
    """
    Prints progress every `print_every` completed tasks.
    Uses overall elapsed/done rate which is naturally smooth.
    """
    def __init__(self, n_total, label='', print_every=None):
        self.n_total     = n_total
        self.n_done      = 0
        self.t_start     = time.perf_counter()
        self.label       = label
        self.print_every = print_every or max(1, n_total // 50)
        self._lock       = threading.Lock()

    def __call__(self, result):
        with self._lock:
            self.n_done += 1
            if self.n_done % self.print_every != 0 and self.n_done != self.n_total:
                return
            elapsed = time.perf_counter() - self.t_start
            rate    = elapsed / self.n_done
            eta     = rate * (self.n_total - self.n_done) / 60
            pct     = 100 * self.n_done / self.n_total
            print(f"  {self.label} [{self.n_done}/{self.n_total}  {pct:.0f}%]"
                  f"  {elapsed/60:.1f} min elapsed,  ETC: {eta:.1f} min"
                  f"  ({rate:.1f} s/task)",
                  flush=True)

# —— Plot helpers
def make_cyclic_cmap(name='cyclic_RdBu'):
    """
    Cyclic colormap for polarization angle psi.
    Goes red -> white -> blue -> white -> red, 
    so psi=0 is white (consistent with RdBu_r at 0),
    and the ±90° endpoints meet continuously.
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Sample RdBu_r: 0=blue, 0.5=white, 1=red
    base = plt.cm.RdBu_r
    # Build: white(0°) -> red(45°) -> white(90°/-90°) -> blue(-45°) -> white(0°)
    # In [0,1] colormap coordinates: 0->0.5, 0.25->1.0, 0.5->0.5, 0.75->0.0, 1.0->0.5
    t = np.linspace(0, 1, 512)
    # Map t to a ping-pong through RdBu_r
    u = 0.5 + 0.5 * np.sin(2 * np.pi * t)  # oscillates 0.5->1->0.5->0->0.5
    colors = base(u)
    return LinearSegmentedColormap.from_list(name, colors)

def _get_panels(preset: str, panels_override=None,
                S1max=1.0, S3max=1.0, CDmax=1.0) -> list:
    """
    Return list of (col, label, cmap, vmin, vmax) for the requested preset.
    panels_override, if provided, is returned as-is (allows fully custom panels).
 
    Available presets:
        'crsbr'   — Rss, Rpp, R_S1s, R_S3s, R_CD  (reflection, polarization)
        'lc'      — R_total, T_total, R_S1s, R_S1p, R_S3s, R_S3p, R_CD
        'dbr'     — R_total, T_total
        'full'    — everything: Rss/pp/sp/ps + Stokes + circular
        'totalstokes' — Total stokes parameters from reflection
        None      — same as 'crsbr'
    """
    if panels_override is not None:
        return panels_override
    cyclic_RdBu = make_cyclic_cmap()
 
    match preset:
        case 'totalstokes' | None:
            return [
                ('T_S0u', r'$S_{0}^{(sp)}$', 'magma', 0, 1),
                ('T_S1un', r'$S_{1}^{(sp)}$', 'RdBu_r', -S1max, S1max),
                ('T_S2un', r'$S_{2}^{(sp)}$', 'RdBu_r', -S1max, S1max),
                ('T_DCP_u', r'$S_{3}^{(sp)}$', 'RdBu_r', -CDmax, CDmax),
                ('T_psi_s', r'$\psi^{(s)}$ (°)', cyclic_RdBu, -90, 90),
                ('T_chi_s', r'$\chi^{(s)}$ (°)', 'PiYG', -45, 45),
            ]
        case 'psichiTE':
            return [
                ('T_psi_s', r'$\psi^{(s)}$ (°)', cyclic_RdBu, -90, 90),
                ('T_chi_s', r'$\chi^{(s)}$ (°)', 'PiYG', -45, 45),
            ]
        case 'res2':
            return [
                ('T_ss', r'$T_{ss}$', 'magma', 0, 1),
                # ('T_sp', r'$T_{sp}$', 'magma', 0, 1),
                # ('T_ps', r'$T_{ps}$', 'magma', 0, 1),
                ('T_pp', r'$T_{pp}$', 'magma', 0, 1),
                # ('T_S0s', r'$S_{0}^{(s)}$', 'magma', 0, 1),
                # ('T_S1s', r'$S_{1}^{(s)}$', 'RdBu_r', -S1max, S1max),
                ('T_S1un', r'$S_{1}^{(sp)}$', 'RdBu_r', -S1max, S1max),
                # ('T_S2s', r'$S_{2}^{(s)}$', 'RdBu_r', -S1max, S1max),
                # ('T_S3s', r'$S_{3}^{(s)}$', 'RdBu_r', -CDmax, CDmax),
                ('T_S3un', r'$S_{3}^{(sp)}$', 'PiYG', -CDmax, CDmax),
                ]
        case 'stokesS':
            return [
                ('R_S0s', r'$S_{0}^{(s)}$', 'magma', 0, 1),
                ('R_S1s', r'$S_{1}^{(s)}$', 'RdBu_r', -S1max, S1max),
                ('R_S2s', r'$S_{2}^{(s)}$', 'RdBu_r', -S1max, S1max),
                ('R_S3s', r'$S_{3}^{(s)}$', 'RdBu_r', -CDmax, CDmax),
                ('R_psi_s', r'$\psi^{(s)}$ (°)', cyclic_RdBu, -90, 90),
                ('R_chi_s', r'$\chi^{(s)}$ (°)', 'PiYG', -45, 45),
            ]
        case 'stokesP':
            return [
                ('R_S0p', r'$S_{0}^{(p)}$', 'magma', 0, 1),
                ('R_S1p', r'$S_{1}^{(p)}$', 'RdBu_r', -S1max, S1max),
                ('R_S2p', r'$S_{2}^{(p)}$', 'RdBu_r', -S1max, S1max),
                ('R_S3p', r'$S_{3}^{(p)}$', 'RdBu_r', -CDmax, CDmax),
                ('R_psi_p', r'$\psi^{(p)}$ (°)', cyclic_RdBu, -90, 90),
                ('R_chi_p', r'$\chi^{(p)}$ (°)', 'PiYG', -45, 45),
            ]
        case 'lc':
            return [
                ('R_total', r'$R_{tot}$',            'magma',  0,       1      ),
                ('T_total', r'$T_{tot}$',            'magma',  0,       1      ),
                ('R_S1s',   r'$S_1^{(s)}$ TE-TM',   'RdBu_r', -S1max,  S1max  ),
                ('R_S1p',   r'$S_1^{(p)}$ TE-TM',   'RdBu_r', -S1max,  S1max  ),
                ('R_S3s',   r'$S_3^{(s)}$ circ',    'PiYG',   -S3max,  S3max  ),
                ('R_S3p',   r'$S_3^{(p)}$ circ',    'PiYG',   -S3max,  S3max  ),
                ('R_CD',    r'$|RR|^2-|LL|^2$',      'RdBu_r', -CDmax,  CDmax  ),
            ]
        case 'dbr':
            return [
                ('R_S0u', r'$R_{tot}$', 'magma', 0, 1),
                ('T_S0u', r'$T_{tot}$', 'magma', 0, 1),
            ]
        case 'full':
            return [
                ('R_ss',   r'$R_{ss}$',             'magma',  0,       1      ),
                ('R_pp',   r'$R_{pp}$',             'magma',  0,       1      ),
                ('R_sp',   r'$R_{sp}$',             'magma',  0,       1      ),
                ('R_ps',   r'$R_{ps}$',             'magma',  0,       1      ),
                ('R_S1s',  r'$S_1^{(s)}$',          'RdBu_r', -S1max,  S1max  ),
                ('R_S2s',  r'$S_2^{(s)}$',          'RdBu_r', -S1max,  S1max  ),
                ('R_S3s',  r'$S_3^{(s)}$',          'PiYG',   -S3max,  S3max  ),
                ('R_S1p',  r'$S_1^{(p)}$',          'RdBu_r', -S1max,  S1max  ),
                ('R_S3p',  r'$S_3^{(p)}$',          'PiYG',   -S3max,  S3max  ),
                ('R_RR',   r'$R_{RR}$',             'magma',  0,       1      ),
                ('R_LL',   r'$R_{LL}$',             'magma',  0,       1      ),
                ('R_CD',   r'$|RR|^2-|LL|^2$',      'RdBu_r', -CDmax,  CDmax  ),
            ]
        case _:
            raise ValueError(f"Unknown preset '{preset}'. "
                             f"Choose from: 'totalstokes', 'crsbr', 'lc', 'dbr', 'full', or pass panels=[(col, lbl, cmap, vmin, vmax), ...]")

# —— Study 0 plot ——————————————————————————————————————————————————————————————

def plot_study0(df, conf: object,
                quantities=None, save_fig=False) -> plt.figure:
    """
    Plot R, T, A spectra from Study 0.

    Parameters
    ----------
    quantities : list of (col, label, color) to plot.
                 Defaults to R_s, R_p, T_s, T_p.
    """
    layers = conf.layers
    a = conf.lattice_const

    if quantities is None:
        quantities = [
            ('R_s', r'$R_s$',  'steelblue',  '-'),
            ('R_p', r'$R_p$',  'steelblue',  '--'),
            ('T_s', r'$T_s$',  'crimson',    '-'),
            ('T_p', r'$T_p$',  'crimson',    '--'),
            ('A_s', r'$A_s$',  'gray',       '-'),
        ]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), layout='constrained')
    fig.suptitle(stack_title(layers, a, extras='Normal incidence R/T/A'), fontsize=9)

    # Left: wavelength axis
    for col, lbl, color, ls in quantities:
        if col in df.columns:
            ax1.plot(df['lambda0'], df[col], label=lbl, color=color, ls=ls)
    ax1.set(xlabel='Wavelength (nm)', ylabel='Fraction', ylim=(-0.02, 1.02))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # # Right: energy axis
    # for col, lbl, color, ls in quantities:
    #     if col in df.columns:
    #         ax2.plot(df['energy'], df[col], label=lbl, color=color, ls=ls)
    # ax2.set(xlabel='Energy (eV)', ylabel='Fraction', ylim=(-0.02, 1.02))
    # ax2.legend(fontsize=8)
    # ax2.grid(True, alpha=0.2)

    if save_fig:
        fname = conf.output_dir / f"study0_{make_stack_slug(conf)}.pdf"
        fig.savefig(fname, format='pdf', bbox_inches='tight')
        print(f"Saved {fname}")
    return fig

# ── Study 1 plot ──────────────────────────────────────────────────────────────

def plot_study1(df, conf,
                rot_deg=None,
                azim_vals=None,
                x_col='elev',
                elev_range=None,
                preset='crsbr',
                panels=None,
                S1max=1.0, S3max=1.0, CDmax=1.0,
                save_fig=False):
    """
    E vs k dispersion plots.
 
    Parameters
    ----------
    df         : DataFrame from run_study1
    conf       : RCWAConfig
    rot_deg    : which global_rot slice to plot (None = use conf.global_rot)
    azim_vals  : list of azimuths to show as rows (None = use conf.study1.azim_vals)
    x_col      : 'elev' or 'k_parallel'
    elev_range : (min_deg, max_deg) to crop x axis
    preset     : 'crsbr' | 'lc' | 'dbr' | 'full'  (ignored if panels provided)
    panels     : custom list of (col, label, cmap, vmin, vmax); overrides preset
    S1max/S3max/CDmax : colorbar limits for Stokes / CD panels
    save_fig   : save PDF to conf.output_dir
    """
    layers   = conf.layers
    a        = conf.lattice_const
    rot_deg  = rot_deg  if rot_deg  is not None else (conf.global_rot or 0.0)
    azim_vals = azim_vals if azim_vals is not None else conf.study1.azim_vals
 
    # Compute derived quantities on-demand
    df_ext = _add_obs(df)
 
    panel_list = _get_panels(preset, panels, S1max, S3max, CDmax)
 
    rot_filter = round(float(rot_deg), 2)
    df_rta = df[df_ext['rot'].round(2) == rot_filter]
    wl     = sorted(df_rta['lambda0'].unique())
 
    nrows, ncols = len(azim_vals), len(panel_list)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 5 * nrows),
                              layout='constrained',
                              sharex=True, sharey=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]
 
    rot_tag = f'{rot_deg:.0f}°'
    if preset == 'dbr':
        title   = f'{_make_unit_cell_title(layers, a)}\nDBR reflection & transmission'
        fig.suptitle(title, fontsize=12)
    else:
        title   = f'{_make_unit_cell_title(layers, a)}\n$\phi_z$ = {rot_tag}'
        fig.suptitle(title, fontsize=18)
   
 
    for i, az in enumerate(azim_vals):
        df_s = df_rta[df_rta['azim'] == az].copy()
        for j, (col, lbl, cmap, vmin, vmax) in enumerate(panel_list):
            ax = axes[i, j]
 
            if col not in df_s.columns:
                ax.text(0.5, 0.5, f'"{col}"\nnot in df',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='gray')
                continue
 
            pivot = df_s.pivot_table(index='energy', columns='elev', values=col)
            pivot = pivot.sort_index().sort_index(axis=1)
 
            if pivot.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
 
            ec = pivot.columns.values
            if x_col == 'k_parallel':
                lm  = np.median(df_s['lambda0'].values)
                kc  = (2 * np.pi / lm) * np.sin(np.radians(ec)) * 1000
                ext = [kc.min(), kc.max(),
                       pivot.index.min(), pivot.index.max()]
                xl  = r'$k_\parallel$ ($\mu$m$^{-1}$)'
            else:
                ext = [ec.min(), ec.max(),
                       pivot.index.min(), pivot.index.max()]
                xl  = r'$\theta$ (deg)'
 
            im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                           interpolation='bicubic', extent=ext,
                           cmap=cmap, vmin=vmin, vmax=vmax)
 
            if elev_range is not None:
                ax.set_xlim(elev_range[0], elev_range[1])
            if i == 0:
                ax.set_title(lbl, fontsize=14, y=1.15)
            if j == 0:
                if preset == 'dbr':
                    ax.set_ylabel(f'Energy (eV)')
                else:
                    ax.set_ylabel(f'azim={az:.0f}°\nEnergy (eV)')
            if i == nrows - 1:
                ax.set_xlabel(xl, fontsize=12)
            if j == ncols - 1:
                ax2 = ax.twinx()
                ax2.set_ylabel(r'$\lambda$ (nm)')
                ax2.set_ylim(max(wl), min(wl))
            # fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02) 
            if i == 0: 
                if vmin == 0:
                    ticks = [0, 1]
                elif vmin == -1:
                    ticks = [-1, 1]
                elif vmin == -90:
                    ticks = [-90, 90]
                elif vmin == -45:
                    ticks = [-45, 45]
                else:
                    ticks = [vmin, round((vmin+vmax)/2, 2), vmax]
                cax = ax.inset_axes([0, 1.05, 1, 0.05])
                fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, ticks=ticks, cax=cax, location='top', aspect=10)   
 
    if save_fig:
        rot_str = f'_rot{int(rot_deg or 0):02d}'
        fname = conf.output_dir / f'study1_{make_stack_slug(conf)}{rot_str}.pdf'
        fig.savefig(fname, format='pdf', bbox_inches='tight')
        print(f'Saved {fname}')
        print(f'%--'*30)
        print()
 
    return fig

# —— Study 2 plots

def plot_kxky_grid(df, conf,
                   energies_eV=None,
                   preset='crsbr',
                   panels=None,
                   S1max=1.0, S3max=1.0, CDmax=1.0,
                   save_fig=False):
    """
    Grid of kx-ky maps at selected energies.
 
    preset / panels : same as plot_study1.
    """
    layers    = conf.layers
    a         = conf.lattice_const
    rot_deg   = conf.global_rot if conf.global_rot is not None else 0.0
    output_dir = conf.output_dir
 
    df = _add_obs(df)
    panel_list = _get_panels(preset, panels, S1max, S3max, CDmax)
 
    df_p  = df[df['rot'].round(2) == round(rot_deg, 2)]
    all_e = sorted(df_p['energy'].unique())
 
    if energies_eV is None:
        idx = np.linspace(0, len(all_e) - 1, 4, dtype=int)
        energies_eV = [all_e[i] for i in idx]
    else:
        energies_eV = [all_e[np.argmin(np.abs(np.array(all_e) - e))]
                       for e in energies_eV]
 
    kx_all = df_p['kx_nm'].values * 1000
    ky_all = df_p['ky_nm'].values * 1000
    kmax   = max(np.abs(kx_all).max(), np.abs(ky_all).max())
    ki     = np.linspace(-kmax, kmax, 400)
    kxi, kyi  = np.meshgrid(ki, ki)
    disk_mask = kxi**2 + kyi**2 > kmax**2
 
    nrows, ncols = len(energies_eV), len(panel_list)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4 * nrows),
                              layout='constrained',
                              sharex=True, sharey=True)
    if nrows == 1: axes = axes[np.newaxis, :]
    if ncols == 1: axes = axes[:, np.newaxis]
 
    title = stack_title(layers, a,
                        extras=f'rot = {rot_deg:.0f}°  |  kx-ky map')
    fig.suptitle(title, fontsize=9)
 
    for i, e_sel in enumerate(energies_eV):
        df_e = df_p[np.isclose(df_p['energy'], e_sel, atol=1e-4)]
        kx   = df_e['kx_nm'].values * 1000
        ky   = df_e['ky_nm'].values * 1000
        ls   = 1239.8 / e_sel
 
        for j, (col, lbl, cmap, vmin, vmax) in enumerate(panel_list):
            ax = axes[i, j]
 
            if col not in df_e.columns:
                ax.text(0.5, 0.5, f'"{col}"\nnot in df',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='gray')
                continue
 
            zi = griddata((kx, ky), df_e[col].values,
                          (kxi, kyi), method='linear')
            zi[disk_mask] = np.nan
            im = ax.imshow(zi, extent=[-kmax, kmax, -kmax, kmax],
                           origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation='bilinear', aspect='equal')
            if i == 0:
                ax.set_title(lbl, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'E={e_sel:.3f} eV\n'
                              f'λ={ls:.0f} nm\n'
                              r'$k_y$ ($\mu$m$^{-1}$)')
            if i == nrows - 1:
                ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)')
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.axvline(0, color='k', lw=0.5, alpha=0.3)
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
 
    if save_fig:
        out = conf.output_dir / (make_study_fname(2, conf) + '_grid.pdf')
        fig.savefig(out, format='pdf', bbox_inches='tight')
        print(f'Saved --> {out}')
 
    return fig

def plot_polarization_kxky(df, conf, energy_eV,
                            input_pol='s',
                            save_fig=False):
    """
    kx-ky maps of polarization orientation angle ψ and ellipticity χ.
    C-points appear as phase singularities in ψ and circular spots in χ.
 
    input_pol : 's' or 'p' — which input polarisation to show
    """
    layers    = conf.layers
    a         = conf.lattice_const
    rot_deg   = conf.global_rot if conf.global_rot is not None else 0.0
    output_dir = conf.output_dir
 
    df = _add_obs(df)
 
    df_p     = df[df['rot'].round(2) == round(rot_deg, 2)]
    energies = df_p['energy'].unique()
    e_sel    = energies[np.argmin(np.abs(energies - energy_eV))]
    df_e     = df_p[np.isclose(df_p['energy'], e_sel, atol=1e-4)].copy()
 
    # Choose Stokes columns for the requested input polarisation
    pol = input_pol.lower()
    s0col = f'R_S0{pol}'; s1col = f'R_S1{pol}'
    s2col = f'R_S2{pol}'; s3col = f'R_S3{pol}'
 
    s0 = df_e[s0col].values.copy()
    s0[s0 < 1e-6] = np.nan
    s1n = df_e[s1col].values / s0
    s2n = df_e[s2col].values / s0
    s3n = df_e[s3col].values / s0
 
    df_e['psi_deg'] = np.degrees(0.5 * np.arctan2(s2n, s1n))
    df_e['chi_deg'] = np.degrees(0.5 * np.arcsin(np.clip(s3n, -1, 1)))
 
    kx   = df_e['kx_nm'].values * 1000
    ky   = df_e['ky_nm'].values * 1000
    kmax = max(np.abs(kx).max(), np.abs(ky).max())
    ki   = np.linspace(-kmax, kmax, 400)
    kxi, kyi  = np.meshgrid(ki, ki)
    disk_mask = kxi**2 + kyi**2 > kmax**2
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    title = stack_title(layers, a,
                        extras=(f'Polarisation map ({pol}-in)  |  '
                                f'E={e_sel:.3f} eV  λ={1239.8/e_sel:.0f} nm  '
                                f'rot={rot_deg:.0f}°'))
    fig.suptitle(title, fontsize=9)
 
    for ax, qty, cmap, vmin, vmax, lbl in [
        (ax1, 'psi_deg', 'twilight_shifted', -90, 90,
         r'$\psi$ orientation (deg)'),
        (ax2, 'chi_deg', 'PiYG', -45, 45,
         r'$\chi$ ellipticity (deg)'),
    ]:
        zi = griddata((kx, ky), df_e[qty].values,
                      (kxi, kyi), method='linear')
        zi[disk_mask] = np.nan
        im = ax.imshow(zi, extent=[-kmax, kmax, -kmax, kmax],
                       origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='bilinear', aspect='equal')
        ax.set(title=lbl,
               xlabel=r'$k_x$ ($\mu$m$^{-1}$)',
               ylabel=r'$k_y$ ($\mu$m$^{-1}$)')
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        fig.colorbar(im, ax=ax, label=lbl)
 
    if save_fig:
        fname = output_dir / (f'study2_polmap_{pol}in_rot{int(round(rot_deg)):02d}'
                              f'_E{e_sel:.3f}eV_A{a:.0f}nm.pdf')
        fig.savefig(fname, format='pdf', bbox_inches='tight')
        print(f'Saved {fname}')
 
    return fig

def animate_kxky(df, conf,
                 rot_deg=None,
                 preset='crsbr',
                 panels=None,
                 S1max=1.0, S3max=1.0, CDmax=1.0,
                 output_path=None, fps=4, dpi=150):
    """
    Compile kx-ky maps at each wavelength into an mp4.
 
    preset / panels : same as plot_study1.
    """
    layers    = conf.layers
    a         = conf.lattice_const
    rot_deg   = rot_deg if rot_deg is not None else (conf.global_rot or 0.0)
 
    df = _add_obs(df)
    panel_list = _get_panels(preset, panels, S1max, S3max, CDmax)
 
    if output_path is None:
        output_path = conf.output_dir / (make_study_fname(2, conf) + f'_kxky_preset-{preset}.mp4')
 
    df_p = df[df['rot'].round(2) == round(rot_deg, 2)].copy()
    wavelengths = sorted(df_p['lambda0'].unique())
 
    df_p['kx_n'] = (np.sin(np.radians(df_p['elev']))
                    * np.cos(np.radians(df_p['azim'])))
    df_p['ky_n'] = (np.sin(np.radians(df_p['elev']))
                    * np.sin(np.radians(df_p['azim'])))
    kmax = np.sin(np.radians(conf.study2.elev_max))
 
    ki = np.linspace(-kmax, kmax, 300)
    kxi, kyi  = np.meshgrid(ki, ki)
    disk_mask = kxi**2 + kyi**2 > kmax**2
 
    print(f'Pre-computing {len(wavelengths)} frames...', end=' ', flush=True)
    frames_data = []
    for lam in wavelengths:
        df_e = df_p[np.isclose(df_p['lambda0'], lam, atol=0.1)]
        kx   = df_e['kx_n'].values
        ky   = df_e['ky_n'].values
        frame = {}
        for col, *_ in panel_list:
            if col in df_e.columns:
                zi = griddata((kx, ky), df_e[col].values,
                              (kxi, kyi), method='linear')
                zi[disk_mask] = np.nan
                frame[col] = zi
        frames_data.append(frame)
    print('done.')
 
    ncols = len(panel_list)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5),
                              constrained_layout=True)
    if ncols == 1:
        axes = [axes]
 
    images = []
    for j, (col, lbl, cmap, vmin, vmax) in enumerate(panel_list):
        ax = axes[j]
        im = ax.imshow(frames_data[0].get(col, np.zeros((300, 300))),
                       extent=[-kmax, kmax, -kmax, kmax],
                       origin='lower', aspect='equal',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='bilinear')
        ax.set_title(lbl, fontsize=12)
        ax.set_xlabel(r'$\sin\theta\cos\phi$')
        if j == 0:
            ax.set_ylabel(r'$\sin\theta\sin\phi$')
        else:
            ax.set_yticks([])
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        fig.colorbar(im, ax=ax, shrink=0.85, fraction=0.046, pad=0.04)
        images.append(im)
 
    stack_lbl = _get_patterned_layer_info(layers)
    title_obj = fig.suptitle('', fontsize=14)
 
    def update(frame):
        lam = wavelengths[frame]
        for j, (col, *_) in enumerate(panel_list):
            if col in frames_data[frame]:
                images[j].set_data(frames_data[frame][col])
        title_obj.set_text(
            rf"""{stack_lbl}  |  P = {a:.0f} nm  |  $\phi_z$ = {rot_deg:.0f}°
$\lambda$ = {lam:.0f} nm  $\longleftrightarrow$  E = {1239.8/lam:.3f} eV"""
        )
        return images + [title_obj]
 
    ani = animation.FuncAnimation(fig, update, frames=len(wavelengths),
                                   interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f'Saved {len(wavelengths)} frames -> {output_path.name}')
    return output_path

def animate_polarization(df, conf,
                          rot_deg=None,
                          input_pol='s',
                          output_path=None, fps=4, dpi=150):
    """
    Compile ψ (orientation angle) and χ (ellipticity) kx-ky maps
    at each wavelength into an mp4.

    C-points appear as phase singularities in ψ and circular spots in χ.

    input_pol : 's' or 'p' — which input polarisation to animate
    """
    layers   = conf.layers
    a        = conf.lattice_const
    rot_deg  = rot_deg if rot_deg is not None else (conf.global_rot or 0.0)

    df = _add_obs(df)

    if output_path is None:
        pol_tag = input_pol.lower()
        output_path = conf.output_dir / (
            make_study_fname(2, conf) + f'_polmap_{pol_tag}in.mp4')

    df_p = df[df['rot'].round(2) == round(rot_deg, 2)].copy()
    wavelengths = sorted(df_p['lambda0'].unique())

    # Normalized angular coordinates
    df_p['kx_n'] = (np.sin(np.radians(df_p['elev']))
                    * np.cos(np.radians(df_p['azim'])))
    df_p['ky_n'] = (np.sin(np.radians(df_p['elev']))
                    * np.sin(np.radians(df_p['azim'])))
    kmax = np.sin(np.radians(conf.study2.elev_max))

    ki = np.linspace(-kmax, kmax, 300)
    kxi, kyi  = np.meshgrid(ki, ki)
    disk_mask = kxi**2 + kyi**2 > kmax**2

    # Choose Stokes columns for requested input polarisation
    pol   = input_pol.lower()
    s0col = f'R_S0{pol}'
    s1col = f'R_S1{pol}'
    s2col = f'R_S2{pol}'
    s3col = f'R_S3{pol}'

    print(f'Pre-computing {len(wavelengths)} frames...', end=' ', flush=True)
    frames_data = []
    for lam in wavelengths:
        df_e = df_p[np.isclose(df_p['lambda0'], lam, atol=0.1)].copy()

        s0 = df_e[s0col].values.copy()
        s0[s0 < 1e-6] = np.nan
        s1n = df_e[s1col].values / s0
        s2n = df_e[s2col].values / s0
        s3n = df_e[s3col].values / s0

        psi = np.degrees(0.5 * np.arctan2(s2n, s1n))
        chi = np.degrees(0.5 * np.arcsin(np.clip(s3n, -1, 1)))

        kx = df_e['kx_n'].values
        ky = df_e['ky_n'].values

        zi_psi = griddata((kx, ky), psi, (kxi, kyi), method='linear')
        zi_chi = griddata((kx, ky), chi, (kxi, kyi), method='linear')
        zi_psi[disk_mask] = np.nan
        zi_chi[disk_mask] = np.nan
        frames_data.append({'psi': zi_psi, 'chi': zi_chi})
    print('done.')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    constrained_layout=True)
    im1 = ax1.imshow(frames_data[0]['psi'],
                     extent=[-kmax, kmax, -kmax, kmax],
                     origin='lower', aspect='equal',
                     cmap='twilight_shifted', vmin=-90, vmax=90,
                     interpolation='none')
    im2 = ax2.imshow(frames_data[0]['chi'],
                     extent=[-kmax, kmax, -kmax, kmax],
                     origin='lower', aspect='equal',
                     cmap='PiYG', vmin=-45, vmax=45,
                     interpolation='none')

    for ax, lbl in [(ax1, r'$\psi$ orientation (°)'),
                    (ax2, r'$\chi$ ellipticity (°)')]:
        ax.set_xlabel(r'$\sin\theta\cos\phi$')
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.set_title(lbl)
    ax1.set_ylabel(r'$\sin\theta\sin\phi$')
    ax2.set_yticks([])
    fig.colorbar(im1, ax=ax1, shrink=0.85, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, shrink=0.85, fraction=0.046, pad=0.04)

    stack_lbl = _get_patterned_layer_info(layers)
    title_obj = fig.suptitle('', fontsize=12)

    def update(frame):
        lam = wavelengths[frame]
        im1.set_data(frames_data[frame]['psi'])
        im2.set_data(frames_data[frame]['chi'])
        title_obj.set_text(
            rf"""{stack_lbl}  |  P = {a:.0f} nm  |  rot = {rot_deg:.0f}°  |  {pol}-in
$\lambda$ = {lam:.0f} nm  $\longleftrightarrow$  E = {1239.8/lam:.3f} eV"""
        )
        return [im1, im2, title_obj]

    ani = animation.FuncAnimation(fig, update, frames=len(wavelengths),
                                   interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f'Saved {len(wavelengths)} frames -> {output_path.name}')
    return output_path

# —— Study 3 plots

def plot_mode_scan(conf: object, lam_real, t_ss_vals, layers=None, a=None,
                   elev_deg=0.0, azim_deg=0.0, eps_imag=None,
                   save_fig=False):
    """
    Plot |t_ss| and |1/t_ss| from a single-point pole scan.
    Peaks in |1/t_ss| mark mode energies.
    """
    if layers is None: layers = conf.layers
    if a      is None: a      = conf.lattice_const
    output_dir = conf.output_dir

    energy = 1239.8 / lam_real
    inv    = 1.0 / (np.abs(t_ss_vals) + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')

    extras = (f'elev={elev_deg:.1f} deg  azim={azim_deg:.0f} deg'
              + (f'  eps_imag={eps_imag}' if eps_imag is not None else ''))
    fig.suptitle(stack_title(layers, a, extras=extras), fontsize=9)

    ax1.plot(energy, np.abs(t_ss_vals), color='steelblue')
    ax1.set(xlabel='Energy (eV)', ylabel='|t_ss|', title='Transmission amplitude')
    ax1.grid(True, alpha=0.2)

    ax2.plot(energy, inv, color='crimson')
    ax2.set(xlabel='Energy (eV)', ylabel='|1/t_ss|',
            title='S-matrix poles (mode energies)')
    ax2.grid(True, alpha=0.2)

    # Mark peaks in |1/t_ss|
    from scipy.signal import find_peaks
    peaks, props = find_peaks(inv, prominence=inv.max()*0.1)
    for pk in peaks:
        ax2.axvline(energy[pk], color='k', lw=0.8, ls='--', alpha=0.6)
        ax2.text(energy[pk], inv[pk]*1.02,
                 f'{energy[pk]:.3f} eV\n({lam_real[pk]:.0f} nm)',
                 ha='center', va='bottom', fontsize=7)

    if save_fig:
        fname = output_dir / f"study3_singlescan_elev{elev_deg:.0f}_A{a:.0f}nm.pdf"
        fig.savefig(fname, format='pdf', bbox_inches='tight')
        print(f"Saved {fname}")

    return fig

def plot_study3(conf, lam_real, elev_vals, pole_map,
                azim_deg=0.0, eps_imag=None,
                log_scale=True, save_fig=False):
    """
    2D dispersion map of S-matrix poles: energy vs angle.
    Bright bands are photonic modes.

    Parameters
    ----------
    log_scale : bool — use log colorscale (recommended; poles are very sharp)
    """
    layers = conf.layers
    a      = conf.lattice_const

    energy   = 1239.8 / lam_real
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

    extras = (f'S-matrix poles  |  azim={azim_deg:.0f}°'
              + (f'  eps_imag={eps_imag}' if eps_imag is not None else ''))
    fig.suptitle(stack_title(layers, a, extras=extras), fontsize=9)

    ext_elev = [elev_vals.min(), elev_vals.max(),
                energy.min(),    energy.max()]

    lam_mid = np.median(lam_real)
    kpar    = (2*np.pi/lam_mid) * np.sin(np.radians(elev_vals)) * 1000
    ext_k   = [kpar.min(), kpar.max(),
                energy.min(), energy.max()]

    if log_scale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(pole_map.min(), 1e-3), vmax=pole_map.max())
    else:
        norm = None

    for ax, ext, xlabel, title in [
        (axes[0], ext_elev, r'$\theta$ (deg)',
         r'$|1/r_{ss}|$ — mode dispersion'),
        (axes[1], ext_k,    r'$k_\parallel$ ($\mu$m$^{-1}$)',
         r'$|1/r_{ss}|$ — mode dispersion'),
    ]:
        im = ax.imshow(pole_map, aspect='auto', origin='lower',
                       extent=ext, cmap='hot', norm=norm,
                       interpolation='bicubic')
        ax.set(xlabel=xlabel, ylabel='Energy (eV)', title=title)
        ticks = [pole_map.min(), pole_map.max()] if not log_scale else None
        fig.colorbar(im, ax=ax, shrink=0.85, ticks=ticks)

    if save_fig:
        fname = conf.output_dir / (make_study_fname(3, conf) + '.pdf')
        fig.savefig(fname, bbox_inches='tight')
        print(f'Saved {fname}')

    return fig
# —— Study 0: Wavelength-resolved reflection

def run_study0(conf: object,
               save_fig: bool = False):
    """Study 0: normal-incidence R/T/A wavelength sweep."""
    layers = conf.layers
    wavelengths = conf.wavelengths
    n_basis = conf.n_basis
    a = conf.lattice_const

    first_layer_name = 'L0'
    last_layer_name  = f'L{len(layers)-1}'
    S  = build_simulation(conf, wavelengths[0])
    rows = []
    t0   = time.perf_counter()
    lam_rng = (wavelengths[0], wavelengths[-1])
    print(f"Study 0: {len(wavelengths)} wavelengths  |  N_BASIS={n_basis}")
    print(f"  {stack_label(layers, a)}")

    for lam in wavelengths:
        update_simulation_materials(S, float(lam), conf)
        S.SetFrequency(1.0 / lam)
        row = {'lambda0': float(lam), 'energy': float(1239.8 / lam)}
        for pol, sa, pa in [('s', 1, 0), ('p', 0, 1)]:
            S.SetExcitationPlanewave(IncidenceAngles=(0.0, 0.0),
                                      sAmplitude=sa, pAmplitude=pa, Order=0)
            fwd_t, _     = S.GetPowerFlux(Layer=last_layer_name,  zOffset=10.0)
            fwd_r, bwd_r = S.GetPowerFlux(Layer=first_layer_name, zOffset=10.0)
            row[f'T_{pol}'] = float(fwd_t.real)
            row[f'R_{pol}'] = float(-bwd_r.real)
            row[f'A_{pol}'] = float(1 - fwd_t.real + bwd_r.real)
        rows.append(row)

    df = pd.DataFrame(rows)
    fname = make_study_fname(0, conf)

    csv_path, _ = save_study(df, conf, fname)

    print(f"Done in {time.perf_counter()-t0:.1f} s")
    print(f"    Saved -> {csv_path.name}")
    return df

# —— Study 1: E vs. k

def _worker_study1(conf: object, elev_deg: float, azim_deg: float):
    """
    One (elev, azim, rot) across all wavelengths.
    layers is passed explicitly so joblib can pickle it for subprocess delivery.
    """
    layers = conf.layers
    wavelengths = conf.wavelengths
    a = conf.lattice_const
    n_basis = conf.n_basis

    rows = []
    S = build_simulation(conf, float(wavelengths[0]))
    for lam in wavelengths:
        update_simulation_materials(S, float(lam), conf)
        S.SetFrequency(1.0 / lam)
        T, R  = get_jones_matrices(S, elev_deg, azim_deg, conf)
        obs   = compute_observables(T, R)
        kpar  = (2*np.pi/lam) * np.sin(np.radians(elev_deg))
        row   = {
            'rot':        conf.global_rot if conf.global_rot is not None else 0.0,
            'elev':       elev_deg,
            'azim':       azim_deg,
            'lambda0':    lam,
            'energy':     1239.8 / lam,
            'k_parallel': kpar,
        }
        row.update(obs)
        rows.append(row)
    return rows

def run_study1(conf: object, print_every: int = None):

    layers = conf.layers
    azim_vals = conf.study1.azim_vals
    elev_vals = conf.study1.elev_vals 
    wavelengths = conf.wavelengths
    n_jobs = conf.n_jobs
    a = conf.lattice_const
    n_basis = conf.n_basis
    output_dir = conf.output_dir

    t0    = time.perf_counter()
    pairs = list(itertools.product(azim_vals, elev_vals))
    print(f"Study 1: {len(pairs)} pairs  |  {n_jobs} workers")
    print(f"  {stack_label(layers, a)}")

    cb  = ProgressCallback(len(pairs), label='Study 1', print_every=print_every)
    gen = Parallel(n_jobs=n_jobs, verbose=0, return_as='generator')(
        delayed(_worker_study1)(conf, el, az)
        for az, el in pairs
    )
    all_rows = []
    for result in gen:
        cb(result)
        all_rows.extend(result)

    df = pd.DataFrame(all_rows)

    print(f"Study 1 done in {(time.perf_counter()-t0)/60:.1f} min")
    print(f'%--'*30)

    fname = make_study_fname(1, conf)
    csv_path, _ = save_study(df, conf, fname)
    print(f"Data saved as {csv_path.name} ({len(df)} rows)")
    return df

def quick_test_study1(conf: object):
    quick_conf = dataclasses.replace(
        conf,
        lam_step = conf.lam_step * 10,
        n_basis = 25,
        study1 = dataclasses.replace(
            conf.study1, 
            elev_n = 8, 
            azim_vals = [0.0])
    )
    return run_study1(quick_conf)

# —— Study 2: kx vs. ky

def _worker_study2(conf: object, azim_deg: float, elev_deg: float):
    """
    One (azim, elev, phiz) across all wavelengths.
    """
    rows = []
    wavelengths = conf.wavelengths

    S = build_simulation(conf, float(wavelengths[0]))
    for lam in wavelengths:
        update_simulation_materials(S, float(lam), conf)
        S.SetFrequency(1.0 / lam)
        T, R  = get_jones_matrices(S, elev_deg, azim_deg, conf)
        obs   = compute_observables(T, R)
        kpar  = (2*np.pi/lam) * np.sin(np.radians(elev_deg))
        row   = {
            'rot':        conf.global_rot if conf.global_rot is not None else 0.0,
            'elev':       elev_deg,
            'azim':       azim_deg,
            'lambda0':    lam,
            'energy':     1239.8 / lam,
            'k_parallel': kpar,
            'kx_nm':      kpar * np.cos(np.radians(azim_deg)),
            'ky_nm':      kpar * np.sin(np.radians(azim_deg)),
        }
        row.update(obs)
        rows.append(row)
    return rows

def run_study2(conf: object, print_every: int = None):
    
    layers = conf.layers
    pairs = conf.study2.pairs
    n_jobs = conf.n_jobs
    rot_deg = conf.global_rot
    a = conf.lattice_const
    output_dir = conf.output_dir

    t0  = time.perf_counter()
    print(f"Study 2: rot={rot_deg}  |  {len(pairs)} pairs  |  {n_jobs} workers")
    print(f"  {stack_label(layers, a)}")

    cb  = ProgressCallback(len(pairs), label='Study 2', print_every=print_every)
    gen = Parallel(n_jobs=n_jobs, verbose=0, return_as='generator')(
        delayed(_worker_study2)(conf, az, el)
        for az, el in pairs
    )
    all_rows = []
    for result in gen:
        cb(result)
        all_rows.extend(result)

    df    = pd.DataFrame(all_rows)
    fname = make_study_fname(2, conf)
    csv_path, _ = save_study(df, conf, fname)
    print(f"Study 2 done in {(time.perf_counter()-t0)/60:.1f} min  ->  {fname + '.csv'}  ({len(df)} rows)")
    return df

def quick_test_study2(conf: object):
    quick_conf = dataclasses.replace(
        conf,
        lam_step = conf.lam_step * 5,
        study2 = dataclasses.replace(
            conf.study2,
            elev_max = conf.study2.elev_max / 5,
            azim_n_min = 3,
            azim_n_max = 5
            )
    )
    return run_study2(quick_conf)

# —— Study 3: S-matrix pole scan (find modes)

def _worker_study3(conf, elev_deg, azim_deg, lam_real, eps_imag):
    """
    One (elev, azim) across the complex-frequency wavelength scan.
    Returns array of complex r_ss values (reflection, s-input s-output).
    Uses GetAmplitudes with correct azimuth-aware projection.
    """
    n_G = None
    r_ss_vals = np.zeros(len(lam_real), dtype=complex)

    S = build_simulation(conf, float(lam_real[0]))

    for i, lam in enumerate(lam_real):
        update_simulation_materials(S, float(lam), conf)
        S.SetFrequency(complex(1.0/lam, eps_imag/lam))

        phi   = abs(float(elev_deg))
        theta = float(azim_deg) if elev_deg >= 0 else (float(azim_deg) + 180) % 360
        S.SetExcitationPlanewave(IncidenceAngles=(phi, theta),
                                  sAmplitude=1, pAmplitude=0, Order=0)

        if n_G is None:
            n_G = len(S.GetBasisSet())

        _, back = S.GetAmplitudes(Layer='L0', zOffset=0)

        # Azimuth-aware projection onto true s direction
        az = np.radians(float(azim_deg))
        cs, ss = np.cos(az), np.sin(az)
        r_ss_vals[i] = cs*back[0] + ss*back[n_G]

    return r_ss_vals


def run_study3(conf,
               elev_vals=None,
               azim_deg=0.0,
               lam_center_nm=None,
               lam_range_nm=100,
               n_lam=300,
               eps_imag=0.005,
               print_every=None,
               save_fig=True):
    """
    S-matrix pole scan: find photonic mode energies from peaks in |1/r_ss|.

    Sweeps a complex-frequency grid at each (elev, azim) and returns
    a 2D pole map (lam x elev). Peaks in |1/r_ss| mark mode energies.

    Parameters
    ----------
    conf         : RCWAConfig
    elev_vals    : elevation angles to sweep (defaults to conf.study1.elev_vals)
    azim_deg     : azimuthal angle
    lam_center_nm: center wavelength (defaults to median of conf.wavelengths)
    lam_range_nm : total wavelength range to scan
    n_lam        : number of wavelength points
    eps_imag     : imaginary frequency offset (controls pole visibility)
    print_every  : progress print interval
    save_fig     : save PDF of pole map
    """
    if elev_vals    is None: elev_vals    = conf.study1.elev_vals
    if lam_center_nm is None: lam_center_nm = float(np.median(conf.wavelengths))

    lam_real = np.linspace(lam_center_nm - lam_range_nm/2,
                            lam_center_nm + lam_range_nm/2, n_lam)
    n_elev   = len(elev_vals)
    pole_map = np.zeros((n_lam, n_elev))

    t0 = time.perf_counter()
    print(f"Study 3: {n_elev} elev  |  {n_lam} freq pts  |  {conf.n_jobs} workers")
    print(f"  lam {lam_real[0]:.0f}–{lam_real[-1]:.0f} nm  "
          f"eps_imag={eps_imag}  azim={azim_deg:.0f}°")
    print(f"  {stack_label(conf.layers, conf.lattice_const)}")

    cb  = ProgressCallback(n_elev, label='Study 3', print_every=print_every)
    gen = Parallel(n_jobs=conf.n_jobs, verbose=0, return_as='generator')(
        delayed(_worker_study3)(conf, el, azim_deg, lam_real, eps_imag)
        for el in elev_vals
    )
    for k, r_ss_col in enumerate(gen):
        cb(r_ss_col)
        pole_map[:, k] = 1.0 / (np.abs(r_ss_col) + 1e-10)

    print(f"Study 3 done in {(time.perf_counter()-t0)/60:.1f} min")

    # Save
    fname    = make_study_fname(3, conf)
    df_out   = pd.DataFrame(
        pole_map,
        index   = pd.Index(lam_real,  name='lambda_nm'),
        columns = pd.Index(elev_vals, name='elev_deg'),
    )
    csv_path = conf.output_dir / (fname + '.csv')
    df_out.to_csv(csv_path)
    conf.save(conf.output_dir / (fname + '.json'))
    print(f"  Saved -> {csv_path.name}")

    if save_fig:
        fig = plot_study3(conf, lam_real, elev_vals, pole_map,
                          azim_deg=azim_deg, eps_imag=eps_imag)
        fig_path = conf.output_dir / (fname + '.pdf')
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"  Figure -> {fig_path.name}")
        plt.show()

    return lam_real, elev_vals, pole_map