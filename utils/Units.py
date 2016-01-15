#!/usr/bin/python

"""
 module to easily manage units in Python
 To convert units (e.g. cm to km), just multiply the
        number by km/cm
"""
from math import pi, tan

# Time units
second = 1.0
sec = second
millisecond = 1.0e3 * sec
ms = millisecond
microsecond = 1.0e6 * sec
us = microsecond
minute = 60 * sec
hour = 60 * minute
day = 24 * hour
year = 365 * day

#Angular Measure
radians = 1.0
rad = radians
degree = rad * 180.0 / pi
arcminute = 60.0 * degree
arcsecond = 60.0 * arcminute
arcsec = arcsecond


#Length units:
meter = 1.0
m = meter
centimeter = 100.0 * meter
cm = centimeter
millimeter = 1000.0 * meter
mm = millimeter
micron = 1.0e6 * meter
um = micron
nanometer = 1.0e9 * meter
nm = nanometer
angstrom = 1.0e10 * meter
kilometer = 1.0e-3 * meter
km = kilometer
AU = 1.496e11 * meter
lightyear = 3.0e8 * m / sec * year
parsec = 1 * AU / tan(1 * rad / arcsec)


#Mass units
gram = 1.0
g = gram
kilogram = 1e-3 * gram
kg = kilogram
milligram = 1000 * gram
mg = milligram


#Electric charge:
Coulomb = 1.0
esu = 2.998e9 * Coulomb
electroncharge = 1.0 / 1.609e-19 * Coulomb
electron = electroncharge

#Energy units
Joule = 1.0
erg = 1.0e7 * Joule
electronvolt = electron
eV = electronvolt
keV = 1.0e-3 * eV
MeV = 1.0e-6 * eV
GeV = 1.0e-9 * eV
meV = 1.0e3 * eV

#Pressure units
dyne = 1.0 * g * cm / sec ** 2
Pascal = 1.0 * kg * m / sec ** 2
Pa = Pascal
HectoPascal = 0.01 * Pascal
hPa = HectoPascal
inch_Hg = 1. / 3377. * Pa
torr = 1.333 * hPa


#Some constants in cgs
h = 6.62607e-27
k_b = 1.38065e-16
c = 2.99792e+10
G = 6.67259e-8
Rsun = 6.96e10
Msun = 1.99e33
Mjup = 1.899e30
Mearth = 5.976e27


