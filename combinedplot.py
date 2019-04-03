#!/usr/bin/env python3
# Data order: Initial Beam, With Foil, Model With Foil

import sys
import csv
import matplotlib.pyplot as plt

def dataread(name):
    d_inx = []
    d_iny = []
    with open(str(name), 'r') as infile:
        rowdata = csv.reader(infile)
        for row in rowdata:
            d_inx.append(float(row[0]))
            d_iny.append(float(row[1]))
    return d_inx, d_iny

no_foilx, no_foily = dataread(sys.argv[1])
w_foilx, w_foily = dataread(sys.argv[2])
mw_foilx, mw_foily = dataread(sys.argv[3])

plt.figure()
plt.plot(no_foilx, no_foily, label="Experiment: Initial Beam")
plt.plot(w_foilx, w_foily, label="Experiment: 4 Gold Layers")
plt.plot(mw_foilx, mw_foily, label="Model: 4 Gold Layers")
plt.legend()
plt.xlabel("Angle (Degrees)")
plt.ylabel("Counts")
plt.title("Counts vs. Angle")
plt.savefig('CombinedCountsVAngles')
plt.close()


