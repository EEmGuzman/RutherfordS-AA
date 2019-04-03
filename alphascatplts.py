#!/usr/bin/env python3

# This program will take the raw data file, parse the data, and plot the results.

import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

# Parsing data
time = []
xpixl = []
ypixl = []
with open(str(sys.argv[1]), 'r') as infile:
     rowdata = csv.reader(infile, dialect='excel-tab')
     for row in rowdata:
          time.append(int(row[0]))
          xpixl.append(int(row[1]))
          ypixl.append(int(row[2]))

# Sorting by particle and averaging x position and y pixel position
counter = 0
bad_hit = 0
xsum0 = []
ysum0 = []
xsum1 = []
ysum1 = []
xpixval = []
ypixval = []
for tval in time:
    if tval == 0:
        xsum0.append(xpixl[counter])
        ysum0.append(ypixl[counter])
        counter += 1
        if len(xsum1) != 0:
            difftestx1 = xsum1[-1] - xsum1[0]
            difftesty1 = ysum1[-1] - ysum1[0]
            if abs(difftestx1) <= 10 and abs(difftesty1) <= 10:
                xpixval.append(round(sum(xsum1)/len(xsum1),1))
                ypixval.append(round(sum(ysum1)/len(ysum1),1))
                xsum1 = []
                ysum1 = []
            else:
                print('Combined Hit ends at {}'.format(counter-1))
                xsum1 = []
                ysum1 = []
                bad_hit += 1
    elif tval == 1:
        if len(xsum0) != 0:
            difftestx0 = xsum0[-1] - xsum0[0]
            difftesty0 = ysum0[-1] - ysum0[0]
            if abs(difftestx0) <= 10 and abs(difftesty0) <= 10:
                xpixval.append(round(sum(xsum0)/len(xsum0),1))
                ypixval.append(round(sum(ysum0)/len(ysum0),1))
                xsum0 = []
                ysum0 = []
            else:
                print('Combined Hit ends at {}'.format(counter))
                xsum0 = []
                ysum0 = []
                bad_hit += 1
        xsum1.append(xpixl[counter])
        ysum1.append(ypixl[counter])
        counter += 1

# Ensure final Value not left off
if len(xsum1) != 0:
    difftestx1 = xsum1[-1] - xsum1[0]
    difftesty1 = ysum1[-1] - ysum1[0]
    if abs(difftestx1) <= 10 and abs(difftesty1) <= 10:
        xpixval.append(round(sum(xsum1)/len(xsum1),1))
        ypixval.append(round(sum(ysum1)/len(ysum1),1))
        xsum1 = []
        ysum1 = []
    else:
        print('Combined Hit ends at {}'.format(counter-1))
        xsum1 = []
        ysum1 = []
        bad_hit += 1

if len(xsum0) != 0:
    difftestx0 = xsum0[-1] - xsum0[0]
    difftesty0 = ysum0[-1] - ysum0[0]
    if abs(difftestx0) <= 10 and abs(difftesty0) <= 10:
        xpixval.append(round(sum(xsum0)/len(xsum0),1))
        ypixval.append(round(sum(ysum0)/len(ysum0),1))
        xsum0 = []
        ysum0 = []
    else:
        print('Combined Hit ends at {}'.format(counter))
        xsum0 = []
        ysum0 = []
        bad_hit += 1

print('The number of bad hits is {}'.format(bad_hit))
# Converting from pixel to mm
convfactor = 0.0053 # mm/pixel
mmxpos = [(i * 0.0053) - 2 for i in xpixval]
mmypos = [(i * 0.0053) - 3.5 for i in ypixval]

# Finding center
xmean = np.mean(mmxpos)
ymean = np.mean(mmypos)
print('The center is {},{}'.format(xmean,ymean))
shiftedmmxpos = [i - xmean for i in mmxpos]

# Finding the scattering angle: mmypos is z-axis, y-axis is constant 50mm
deflangles = []
for value in range(len(mmxpos)):
    planeradius = math.sqrt((shiftedmmxpos[value]**2)+(mmypos[value])**2)
    magnitudevect = math.sqrt((shiftedmmxpos[value])**2+(mmypos[value])**2+10**2)
    scatangle = math.degrees(math.asin(math.sqrt(((shiftedmmxpos[value])**2+(mmypos[value])**2))/(magnitudevect)))
    #if planeradius >= 1:
    deflangles.append(scatangle)

# Finding variance
data = np.column_stack((shiftedmmxpos,mmypos)).T
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
with open('WithFoilCurveHist.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(x_smooth, ny_smooth))
# End writer

plt.figure()
plt.plot(x_smooth, y_smooth, c='black')
plt.hist(deflangles, bins=50)
plt.xlabel('Angle (Degrees)')
plt.ylabel('Counts')
plt.title('Counts vs Angle (4 Gold Layers)')
plt.savefig('FinCollWithFoilangles')
plt.close()

# Plot position density
plt.figure()
plt.hist2d(shiftedmmxpos, mmypos, bins=(50, 50), range=[[-3.5,3.5],[-3.5,3.5]], cmap=plt.cm.gist_heat_r)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.colorbar()
plt.title('X-Y Position Dist. of {} events (4 Gold Layers)'.format(len(shiftedmmxpos)))
plt.figtext(0.15, 0.85, "Std Dev x: {}".format(xstd))
plt.figtext(0.15, 0.82, "Std Dev y: {}".format(ystd))
plt.savefig('FinCollWithFoilScat2dhist')
plt.close()
