#!/usr/bin/env python
# coding: utf-8
# Python TOV solver
# Sudhanva Lalit: Date 05/11/2020

''' Information about the code:
This code solves TOV equations for mass radius relations. This also can plot the mass radius curve.

USE: To use the code, here are the steps:
1) Include the file in your main code e.g. import tov_class as tc
2) Load the EoS using the ToV loader, tc.ToV(filename, arraysize)
3) call the solver as tc.ToV.mass_radius(min_pressure, max_pressure)
4) To plot, follow the code in main() on creating the dictionary of inputs

Updates: Version 0.0.1-1
Solves ToV, can only take inputs of pressure, energy density in MeV, baryon density in fm^-3
in ascending order.
'''

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import pylab
from scipy.interpolate import InterpolatedUnivariateSpline

# constants
msol = 1.116e60  # Mass of sun in MeV
Ggrav = 1.324e-42  # Mev^-1*fm
rsol = 2.954e18
rhosol = msol * 3 / (4.0 * np.pi * rsol**3)  # Schwarrzschild radius of the sun in fm


class EoS:
    """ EoS Loader. Interpolates Energy, pressure and number density"""
    alf = 41325.0

    def __init__(self, filename):
        self.file = filename
        self.e_in = np.empty(1000)
        self.p_in = np.empty(1000)
        self.nb_in = np.empty(1000)

    def open_file(self):
        data = np.loadtxt(self.file)
        self.e_in = data[:, 0]
        self.p_in = data[:, 1]
        self.nb_in = data[:, 2]
        print(self.e_in[0], self.p_in[0], self.nb_in[0])
        return self.e_in, self.p_in, self.nb_in

    @staticmethod
    def energy_from_pressure(self, pressure):
        nidx = np.where(self.nb_in == 0.08)
        pcrust = self.p_in[nidx]
        plow = 1e-10
        if pressure < plow:
            return 2.6e-310
        elif pressure < pcrust:
            pres = [self.p_in[i] for i in range(48)]
            eden = [self.e_in[i] for i in range(48)]
            e1 = interp1d(pres, eden, axis=0, kind='linear', fill_value="extrapolate")
            return e1(pressure)
        else:
            e1 = interp1d(self.p_in, self.e_in, axis=0, kind='linear', fill_value="extrapolate")
            return e1(pressure)

    @staticmethod
    def pressure_from_energy(self, energy):
        p1 = interp1d(self.e_in, self.p_in, axis=0, kind='cubic', fill_value="extrapolate")
        return p1(energy)

    @staticmethod
    def baryon_from_energy(self, energy):
        n1 = interp1d(self.e_in, self.nb_in, axis=0, kind='cubic', fill_value='extrapolate')
        return n1(energy)


class ToV(EoS):

    ''' Solves TOV equations and gives data-table, mass-radius plot and max. mass, central pressure
    and central density '''
    alf = 41325.0

    def __init__(self, filename, imax):
        super().__init__(filename)
        self.imax = imax
        self.radius = np.empty(self.imax)
        self.mass = np.empty(self.imax)

    def tov_rhs(self, initial, x):
        pres = initial[0]
        mass = initial[1]
        edn = EoS.energy_from_pressure(self, pres)
        # print("edn", edn, mass, ToV.alf, x)
        # Equations one: pressure, 2: mass
        one = -0.5 * edn * mass * (1.0 + (pres / edn)) * (1. + (4. * np.pi / ToV.alf) * (pres / mass) * x**3) / (x**2 - x * mass)
        two = 4.0 * np.pi * x**2 * edn / ToV.alf
        f = [one, two]
        return f

    def tovsolve(self, pcent, xfinal):
        eden = EoS.energy_from_pressure(self, pcent)
        #print("Eden", pcent, eden)
        dx = 0.001
        x = np.arange(dx, xfinal, dx)
        initial = pcent, 4 * np.pi * dx**3 / (3.0 * ToV.alf)
        psol = odeint(self.tov_rhs, initial, x)
        rstar = 0.
        mstar = 0.
        count = 0
        for i in psol[:, 0]:
            if i > 1.e-7:
                # print("i =", i, count)
                count += 1
                rstar += 2.95 * dx
                mstar = psol[count, 1]
        return rstar, mstar

    def mass_radius(self, pmin, pmax):
        pc = np.zeros(self.imax)
        mass = np.zeros(self.imax)
        radius = np.zeros(self.imax)
        for i in range(self.imax):
            pc[i] = pmin + (pmax - pmin) * i / self.imax
            radius[i], mass[i] = self.tovsolve(pc[i], 10)
        self.radius = radius
        self.mass = mass
        return radius, mass

    # @staticmethod
    def rad14(self):
        n1 = interp1d(self.mass, self.radius, axis=0, kind='cubic', fill_value='extrapolate')
        r14 = n1(1.4)
        # f = interp1d(self.radius, self.mass, axis=0, kind='cubic', fill_value='extrapolate')
        Max_mass = np.max(self.mass)
        nidx = np.where(self.mass == Max_mass)
        Max_radius = self.radius[nidx]
        return print("Radius of 1.4 M_sun star : {} \n Max_mass : {}, Max_radius: {}".format(r14, Max_mass, Max_radius))


def plot(data, **kwargs):
    xl = kwargs['xlabel']
    yl = kwargs['ylabel']
    fl = kwargs['filename']
    ttl = kwargs['title']
    fig = pylab.figure(figsize=(11, 11), dpi=600)
    ax1 = fig.add_subplot(111)
    [ax1.plot(data[0], data[i + 1], label=ttl) for i in range(len(data) - 1)]
    # ax1.plot(data[0], data[1], '-b', label=ttl)

    pylab.xlabel(xl, fontsize=24)
    pylab.ylabel(yl, fontsize=24)
    ax1.tick_params(direction='inout', length=10, width=2, colors='k', grid_color='k', labelsize=24)
    pylab.legend(loc="upper right", fontsize=24)
    pylab.ylim(auto=True)
    pylab.xlim(auto=True)
    pylab.savefig(fl)


def main():
    imax = 1000
    file = ToV("EoS-C.dat", imax)
    file.open_file()
    print(EoS.pressure_from_energy(file, 200), EoS.energy_from_pressure(file, 2.0))
    # print(type(file))
    radius = np.empty(imax)
    mass = np.empty(imax)
    radius, mass = file.mass_radius(2., 1000.)
    data1 = np.array([radius, mass])
    file.rad14()
    data = np.array(data1)
    labels = {'xlabel': 'radius (km)',
              'ylabel': 'Mass (M$_{\odot}$)',
              'filename': 'Mass-Rad.pdf', 'title': 'Mass-Radius'}
    plot(data, **labels)


if __name__ == "__main__":
    main()
