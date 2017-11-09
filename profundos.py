# -*- coding: utf-8 -*-
"""
@author: Davidulicio
"""
import sys  # Me funciona para los tildes
reload(sys)  # si falla la codificación reaplicar reload
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import math
import numpy as np
mag = np.loadtxt('C:\Users\David\Desktop\profundos.dat')
np.sort(mag)
m0=[];m1=[];m2=[];m3=[];m4=[];m5=[];m6=[];m7=[];m8=[];m9=[];m10=[];m11=[]
m12=[];m13=[]
n = np.size(mag)
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
n12 = (np.size(m12) / 44.0)
n13 = (np.size(m13) / 44.0)
a0 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2 + n1 + n0
a1 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2 + n1
a2 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 + n2
a3 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4 + n3 
a4 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 + n4
a5 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6 + n5 
a6 = n13 + n12 + n11 + n10 + n9 + n8 + n7 + n6
a7 = n13 + n12 + n11 + n10 + n9 + n8 + n7
a8 = n13 + n12 + n11 + n10 + n9 + n8
a9 = n13 + n12 + n11 + n10 + n9
a10 = n13 + n12 + n11 + n10
a11 = n13 + n12 + n11
a12 = n13 + n12
a13 = n13
A = np.log10([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13])
M = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
d = np.log10([a8, a9, a10, a11, a12, a13])
M2 = np.array([4.5, 5, 5.5, 6, 6.5, 7])
unos = np.ones_like(M2) 
Gt = np.array([M2, unos])
G = np.matrix.transpose(Gt)
plt.figure(num = 1) # Figura Gunteber-Richter
plt.plot(M, A, 'ro', label = 'Datos Conocidos')
plt.title('Análisis de completitud del catálogo sísmico')
plt.xlabel('Magnitud')
plt.ylabel('Log(N)')
log14 = d[5]
log8 = d[0]
bm = (log14 - log8) / 2.5  # A Y B MAXIMUM LIKEHOOD
am = log8 + 4.5 * bm
arregloult = np.linalg.lstsq(G, d)  # Mínimos cuadrados
b, a = arregloult[0]
print 'b = ' + repr(b)  # b y a dados por mínimos cuadrados
print 'a = ' + repr(a)
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

"Fidelidad de la Gutenberg-Richter"
R = (pearson_def(d, arreglo)) ** 2  # R^2
print 'R^2 = ' + repr(R)
cov = np.linalg.inv(np.matmul(Gt, G))
ob = math.sqrt(cov[0][0])
print 'ob = ' + repr(ob)
oa = math.sqrt(cov[1][1])
print 'oa = ' + repr(oa)
"Distribucion de Poisson"
plt.figure(num = 2)  # Figura Distribucion de Poisson
t = np.linspace(0, 100, 200)
mag = np.linspace(6, 8.5, 6)  # Mags de 6 a 8.5 de 0.5 en 0.5
T = [];P1 = []
for value in mag:
    TD = 1 / (10 ** (a + b * value))  # b ya es negativo por eso +
    P = 1 - np.exp(-t / TD)
    print 'Años para magnitud ' + repr(value) + ': ' + repr(round(TD,1))
    plt.plot(t, P, label='Magnitud ' + repr(value) + ', Años: ' + repr(round(TD,1)))
plt.title('Distribución de Poisson')
plt.xlabel('Años')
plt.ylabel('Probabilidad de ocurrencia')
plt.legend()
plt.show()
"Distribuciones de Weibull"