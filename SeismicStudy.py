# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 16:32:14 2017

@author: David
"""
import sys  # Me funciona para los tildes
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import math
import numpy as np
mag = np.loadtxt('C:\Users\David\Desktop\datos.dat')
mag2 = np.loadtxt('C:\Users\David\Desktop\datoshistoricos.dat')
np.sort(mag)
m0=[];m1=[];m2=[];m3=[];m4=[];m5=[];m6=[];m7=[];m8=[];m9=[];m10=[];m11=[]
m12=[];m13=[];m14=[];m15=[];m16=[];m17=[];m18=[];m19=[]
n = np.size(mag)
n2 = np.size(mag2)
"Ley de Gutenberg-Richter"
for i in range(n):
    trap = mag[i:i+1]
    if trap >= 0 and trap <= 0.5:
        m0.append(trap)
    if trap > 0.5 and trap <= 1:
        m1.append(trap)
    if trap > 1 and trap <= 1.5:
        m2.append(trap)
    if trap > 1.5 and trap <= 2:
        m3.append(trap)
    if trap > 2 and trap <= 2.5:
        m4.append(trap)
    if trap > 2.5 and trap <= 3:
        m5.append(trap)
    if trap > 3 and trap <= 3.5:
        m6.append(trap)
    if trap > 3.5 and trap <= 4:
        m7.append(trap)
    if trap > 4 and trap <= 4.5:
        m8.append(trap)
    if trap > 4.5 and trap <= 5:
        m9.append(trap)
    if trap > 5 and trap <= 5.5:
        m10.append(trap)
    if trap > 5.5 and trap <= 6:
        m11.append(trap)
    if trap > 6 and trap <= 6.5:
        m12.append(trap)
    if trap > 6.5 and trap <= 7:
        m13.append(trap)    
    if trap > 7 and trap <= 7.5:
        m14.append(trap)
for i in range(n2):
    trap1 = mag2[i:i+1]        
    if trap1 > 6.0 and trap1 <= 6.5:
        m15.append(trap1)
    if trap1 > 6.5 and trap1 <= 7:
        m16.append(trap1)        
    if trap1 > 7.0 and trap1 <= 7.5:
        m17.append(trap1)
    if trap1 > 7.5 and trap1 <= 8:
        m18.append(trap1)
    if trap1 > 8 and trap1 <= 8.5:
        m19.append(trap1)
n0 = np.size(m0) / 44.0
n1 = np.size(m1) / 44.0
n2 = np.size(m2) / 44.0
n3 = np.size(m3) / 44.0
n4 = np.size(m4) / 44.0
n5 = np.size(m5) / 44.0
n6 = np.size(m6) / 44.0
n7 = np.size(m7) / 44.0
n8 = np.size(m8) / 44.0
n9 = np.size(m9) / 44.0
n10 = np.size(m10)/ 44.0
n11 = np.size(m11) / 44.0
n12 = (np.size(m12) / 44.0) + (np.size(m15) / 94.0)
n13 = (np.size(m13) / 44.0) + (np.size(m16) / 97.0)
n14 = (np.size(m14) / 44.0) + (np.size(m17) / 95.0)
n18 = np.size(m18) / 221.0
n19 = np.size(m19) / 198.0
a0 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2 + n1 + n0
a1 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2 + n1
a2 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2
a3 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 
a4 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4
a5 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 
a6 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6
a7 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8 + n7
a8 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9 + n8
a9 = n19 + n18 + n14 + n13 + n12 + n11 + n10 + n9
a10 = n19 + n18 + n14 + n13 + n12 + n11 + n10
a11 = n19 + n18 + n14 + n13 + n12 + n11
a12 = n19 + n18 + n14 + n13 + n12
a13 = n19 + n18 + n14 + n13
a14 = n19 + n18 + n14
a15 = n19 + n18
a16 = n19
A = np.log10([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16])
M = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5])
d = np.log10([a8, a9, a10, a11, a12, a13, a14, a15, a16])
M2 = np.array([4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5])
unos = np.ones_like(M2) 
Gt = np.array([M2, unos])
G = np.matrix.transpose(Gt)
plt.figure(num = 1) # Figura Gunteber-Richter
plt.plot(M, A, 'ro', label = 'Datos Conocidos')
plt.title('Análisis de completitud del catálogo sísmico')
plt.xlabel('Magnitud')
plt.ylabel('Log(N)')
log16 = d[7]
log8 = d[0]
bm = (log16 - log8) / 4.0  # A Y B MAXIMUM LIKEHOOD
am = log8 + 4.5 * bm
arregloult = np.linalg.lstsq(G, d)  # Mínimos cuadrados
b, a = arregloult[0]
print b  # b y a dados por mínimos cuadrados
print a
arreglo = b * M2 + a
plt.plot(M2, arreglo, label = 'Ajuste linear')
plt.legend()
"Definicion de algunas funciones"
def average(x):
    "Simplemente el promedio"
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)
R = (pearson_def(d, arreglo)) ** 2  # R^2
print R
cov = np.linalg.inv(np.matmul(Gt, G))
print cov
ob = math.sqrt(cov[0][0])
print ob
oa = math.sqrt(cov[1][1])
print oa
"Distribucion de Poisson"
plt.figure(num = 2)  # Figura Distribucion de Poisson
t = np.linspace(0, 200, 100)
mag = np.linspace(6, 8.5, 6)  # Mags de 6 a 8.5 de 0.5 en 0.5
T = [];P1 = []
for value in mag:
    TD = 10 ** (a - b * value)
    P = 1 - np.exp(-1 * t / TD)
    plt.plot(t, P)