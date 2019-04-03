#!/usr/bin/env python3

# This program will take the scattering csv file data and plot a 2dhist

import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

# Parsing data
xposition = []
yposition = []
with open(str(sys.argv[1]), 'r') as infile:
    rowdata = csv.reader(infile)
    for row in rowdata:
        xposition.append(float(row[0]))
        yposition.append(float(row[1]))
xmean = np.mean(xposition)
ymean = np.mean(yposition)
print('The center should be 0,0 but is {},{}'.format(xmean,ymean))

# Ensuring the model position point does not exceed 4x7 grid
indval2 = 0
for yval in yposition:
    if yval > 2 or yval < -2:
        yposition.pop(indval2)
        xposition.pop(indval2)
    indval2 += 1

for yval in yposition:
    if yval > 2 or yval <-2:
        indval = yposition.index(yval)
        yposition.pop(indval)
        xposition.pop(indval)

# For some reason the first loop over y does not remove all desired values.
# A second application is required. Fix later if time.

for xval in xposition:
    if xval > 3.5 or xval < -3.5:
        indval = xposition.index(xval)
        xposition.pop(indval)
        yposition.pop(indval)

deflangles = []
for value in range(len(xposition)):
    planeradius = math.sqrt((xposition[value]**2)+(yposition[value])**2)
    magnitudevect = math.sqrt((xposition[value])**2+(yposition[value])**2+10**2)
    scatangle = math.degrees(math.asin(math.sqrt(((xposition[value])**2+(yposition[value])**2))/(magnitudevect)))
    #if planeradius >= 1:                                                       
    deflangles.append(scatangle)

# Finding variance                                                              
data = np.column_stack((xposition, yposition)).T
covarmatrix = np.cov(data)
xstd = round(math.sqrt(covarmatrix[0,0]),4)
ystd = round(math.sqrt(covarmatrix[1,1]),4)
print('The max deflection angle is: {}'.format(max(deflangles)))

# Plot count vs angles
counts, bin_edges = np.histogram(deflangles, bins=50)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 5000)
y_sm0 = make_interp_spline(bin_centers, counts, k=5)
y_smooth = y_sm0(x_smooth)
ny_smooth = y_smooth/(max(counts))

# Begin writing out smooth curve data for combined plot
with open('ModelNoFoilCurveHist.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(x_smooth, ny_smooth))

plt.figure()
plt.plot(x_smooth, y_smooth, c='black')
plt.hist(deflangles, bins=50)
plt.xlabel("Angle (Degrees)")
plt.ylabel("Counts")
plt.title("Counts vs Angle (Model Initial Beam)")
plt.savefig('ModelNoFoilScatangles')
plt.close()

# Plotting data
plt.figure()
plt.hist2d(xposition, yposition, bins=(50, 50), range=[[-3.5,3.5],[-3.5,3.5]], cmap=plt.cm.gist_heat_r)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.colorbar()
plt.title("X-Y Position Dist. of {} events (Model Initial Beam)".format(len(xposition)))
plt.figtext(0.15, 0.85, "Std. Dev x: {}".format(xstd))
plt.figtext(0.15, 0.82, "Std. Dev y: {}".format(ystd))
plt.savefig('ModelNoFoilScat2dhist')
plt.close()

