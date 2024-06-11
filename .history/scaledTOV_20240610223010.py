#!/usr/bin/env python
# coding: utf-8
# Python TOV solver
# Sudhanva Lalit: Date 02/14/2023

""" Information about the code:
This code solves TOV equations for mass radius relations. This can also plot the mass-radius curve.

USE: To use the code, here are the steps:
1) Include the file in your main code e.g. import tov_class as tc
2) Load the EoS using the ToV loader, tc.ToV(filename, arraysize)
3) call the solver as tc.ToV.mass_radius(min_pressure, max_pressure)
4) To plot, follow the code in main() on creating the dictionary of inputs

Updates: Version 0.0.1-1
Solves ToV, can only take inputs of pressure (MeV/fm^3), energy density in MeV, baryon density in fm^-3
in ascending order.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import pylab

# constants
r0 = 8.378  # km
m0 = 2.837  # solar mass
alphaG = 5.922e-39  # constant
lambda_n = 2.10e-19  # km
rSun = 2.953  # Schwarzschild Radius of sun in km
eps0 = 1.285e3  # MeV/fm^3
pres0 = 1.285e3  # MeV/fm^3
msol = 1.116e60  # Mass of sun in MeV
Ggrav = 1.324e-42  # Mev^-1*fm
rsol = 2.954e18
# Schwarrzschild radius of the sun in fm
rhosol = msol * 3 / (4.0 * np.pi * rsol**3)


class TOV:
    """Solves TOV equations and gives data-table, mass-radius plot and max. mass, central pressure
    and central density by loading an EoS datafile."""

    def __init__(self, filename, imax):
        self.file = np.loadtxt(filename)
        self.e_in = self.file[:, 0] / eps0  # Scaled Energy density
        self.p_in = self.file[:, 1] / pres0  # Scaled pressure
        self.nb_in = self.file[:, 2]
        self.imax = imax
        self.radius = np.empty(self.imax)
        self.mass = np.empty(self.imax)

    def pressure_from_nb(self, nb):
        """Evaluate pressure from number density using interpolation"""
        p1 = interp1d(
            self.nb_in, self.p_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return p1(nb)

    def energy_from_pressure(self, pressure):
        """Evaluate energy density from pressure using interpolation"""
        plow = 1e-10 / pres0
        if pressure < plow:
            return 2.6e-310
        else:
            e1 = interp1d(
                self.p_in, self.e_in, axis=0, kind="linear", fill_value="extrapolate"
            )
            return e1(pressure)

    def pressure_from_energy(self, energy):
        """Evaluate pressure from energy density using interpolation"""
        p1 = interp1d(
            self.e_in, self.p_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return p1(energy)

    def baryon_from_energy(self, energy):
        """Evaluate number density from energy using interpolation"""
        n1 = interp1d(
            self.e_in, self.nb_in, axis=0, kind="linear", fill_value="extrapolate"
        )
        return n1(energy)

    def tov_rhs(self, initial, x):
        pres, mass = initial
        # mass = initial[1]
        edn = self.energy_from_pressure(pres)
        # Equations one: pressure, 2: mass
        if pres > 0.0:
            one = 0.5 * (edn + pres) * (mass + 3.0 * x**3 * pres) / (x * mass - x**2)
            two = 3.0 * x**2 * edn
        else:
            one = 0.0
            two = 0.0
        return np.asarray([one, two], dtype=np.float64)

    def tovsolve(self, pcent, xfinal):
        dx = 1e-3
        x = np.arange(dx, xfinal, dx)
        # x = np.geomspace(dx, xfinal)
        # print(x)
        initial = pcent, 0.0
        # initial = pcent, 4 * np.pi * dx**3 / (3.0)
        psol = odeint(self.tov_rhs, initial, x, rtol=1e-12, atol=1e-12)
        rstar = 0.0
        mstar = 0.0
        count = 0

        for pval in psol[:, 0]:
            if pval > 1e-10:
                # print("i =", pval, count)
                rstar += dx
                mstar = psol[count, 1]
                count += 1
        # radiusCalc = interp1d(x, psol[:,0], axis=0, kind="linear", fill_value="extrapolate")
        # rstar = fsolve(radiusCalc, 1.0)[0]
        # # print(f"mass = {psol[:,1]} \n pressure = {psol[:,0]}")

        # masInt = interp1d(psol[:, 1], psol[:, 0], axis=0, kind="linear", fill_value="extrapolate")
        # mstar = fsolve(masInt, 0.01)[0]

        # print(rstar*r0, mstar*m0)
        return rstar * r0, mstar * m0

    def mass_radius(self, pmin, pmax):
        pc = np.zeros(self.imax)
        mass = np.zeros(self.imax)
        radius = np.zeros(self.imax)
        pc = np.geomspace(pmin, pmax)
        for i, pcent in enumerate(pc):
            # pc[i] = pmin + (pmax - pmin) * i / self.imax
            radius[i], mass[i] = self.tovsolve(pcent, 10)
        self.radius = np.ma.masked_equal(radius, 0.0).compressed()
        self.mass = np.ma.masked_equal(mass, 0.0).compressed()
        return self.radius, self.mass, pc

    # @staticmethod
    def rad14(self):
        n1 = interp1d(
            self.mass, self.radius, axis=0, kind="linear", fill_value="extrapolate"
        )
        r14 = n1(1.4)
        # f = interp1d(self.radius, self.mass, axis=0, kind='cubic', fill_value='extrapolate')
        Max_mass = np.max(self.mass)
        Max_radius = n1(Max_mass)
        # nidx = np.where(self.mass == Max_mass)
        # Max_radius = self.radius[nidx][0]
        return print(
            "Radius of 1.4 M_sun star : {} \n Max_mass : {}, Max_radius: {}".format(
                r14, Max_mass, Max_radius
            )
        )

    def plot(self, data, **kwargs):
        xl = kwargs["xlabel"]
        yl = kwargs["ylabel"]
        fl = kwargs["filename"]
        ttl = kwargs["title"]
        fig = pylab.figure(figsize=(11, 11), dpi=600)
        ax1 = fig.add_subplot(111)
        [ax1.plot(data[0], data[i + 1], label=ttl) for i in range(len(data) - 1)]
        # ax1.plot(data[0], data[1], '-b', label=ttl)

        pylab.xlabel(xl, fontsize=24)
        pylab.ylabel(yl, fontsize=24)
        ax1.tick_params(
            direction="inout",
            length=10,
            width=2,
            colors="k",
            grid_color="k",
            labelsize=24,
        )
        pylab.legend(loc="upper right", fontsize=24)
        pylab.ylim(auto=True)
        pylab.xlim(auto=True)
        pylab.savefig(fl)


def main():
    imax = 100
    # Replace the filename and run the code
    fileName = "neos.dat"
    file = TOV(fileName, imax)
    print(file.pressure_from_energy(1.0), file.energy_from_pressure(1.0))
    # print(type(file))
    radius = np.empty(imax)
    mass = np.empty(imax)
    radius, mass, pcentral = file.mass_radius(1e-3, 1.5)
    radius = np.ma.masked_equal(radius, 0.0).compressed()
    mass = np.ma.masked_equal(mass, 0.0).compressed()
    # print(f"Radius: {radius} \n Mass: {mass}")
    data1 = np.array([radius, mass])
    file.rad14()
    data = np.array(data1)
    labels = {
        "xlabel": "radius (km)",
        "ylabel": r"Mass (M$_{\odot}$)",
        "filename": "Mass-Rad.pdf",
        "title": "Mass-Radius",
    }
    file.plot(data, **labels)
    labels = {
        "xlabel": "radius (km)",
        "ylabel": r"Pressure ",
        "filename": "Pres-Rad.pdf",
        "title": "Pressure-Radius",
    }
    data2 = np.array([radius, pcentral])
    file.plot(data2, **labels)


if __name__ == "__main__":
    main()
