#########  Matrix Solve  ##############

import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt

from scipy import linalg
from scipy import special

import argparse

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-r1', action = 'store', nargs=1, type=float)
parser.add_argument('-r2', action = 'store', nargs=1, type=float)
parser.add_argument('-l' , action = 'store', nargs=1, type=int)
results = parser.parse_args()
####### constant values #########

r1 = results.r1[0]
r2 = results.r2[0]
l  = results.l[0]

rho_0 = 1.18e03
lam_0 = 4.43e09
mu_0  = 1.59e09 

rho_1 = 11.6e03
lam_1 = 4.23e10
mu_1  = 1.49e10 

rho_2 = 1.30e03
lam_2 = 6.0e05
mu_2  = 4.0e04

#####   Constants   #########

C0L = np.sqrt((lam_0 + 2*mu_0)/rho_0)
C1L = np.sqrt((lam_1 + 2*mu_1)/rho_1)
C2L = np.sqrt((lam_2 + 2*mu_2)/rho_2)

C0T = np.sqrt(mu_0/rho_0)
C1T = np.sqrt(mu_1/rho_1)
C2T = np.sqrt(mu_2/rho_2)

#####   Bessel spherical function for the four arguments ################## 

def bessel(n, x0L):
    return np.sqrt(0.5*np.pi/x0L)*scipy.special.jn(n + 0.5, x0L)

##### Hankel spherical function for the four arguments ####################
    
def hankel(n, x0L):
	return np.sqrt(0.5*np.pi/x0L)*scipy.special.hankel1(n + 0.5, x0L)	

################## Matrixs functions ############################

def Matrix_C(l, r, x1, x2, f):

	m_1_1 = (l/r)*f(l, x1*r) - x1*f(l + 1, x1*r)
	m_1_2 = 0
	m_1_3 = (l*(l+1)/r)*f(l, x2*r)
	m_2_1 = 0
	m_2_2 = f(l, x2*r)
	m_2_3 = 0
	m_3_1 = f(l, x1*r)/r
	m_3_2 = 0
	m_3_3 = (l + 1)*(f(l, x2*r)/r) - x2*f(l + 1, x2*r) 

	return np.array([[m_1_1, m_1_2, m_1_3], [m_2_1, m_2_2, m_2_3], [m_3_1, m_3_2, m_3_3]])

def Matrix_D(l, r, x1, x2, mu, lam, f):
	
	m_1_1 = ((2*mu/r**2)*(l**2 - l - (x1*r)**2) - lam*x1**2)*f(l, x1*r) + 4*mu*(x1/r)*f(l + 1, x1*r)
	m_1_2 = 0
	m_1_3 = (2*mu*l*(l + 1)/r**2)*((l - 1)*f(l, x2*r) - x2*r*f(l + 1, x2*r))
	m_2_1 = 0
	m_2_2 = (mu/r)*((l - 1)*f(l, x2*r) - x2*r*f(l + 1, x2*r))
	m_2_3 = 0
	m_3_1 = (2*mu/r**2)*((l - 1)*f(l, x1*r) - x1*r*f(l + 1, x1*r))
	m_3_2 = 0
	m_3_3 = (2*mu/r**2)*((l**2 - 1 - 0.5*(x2*r)**2)*f(l, x2*r) + x2*r*f(l + 1, x2*r))

	return np.array([[m_1_1, m_1_2, m_1_3], [m_2_1, m_2_2, m_2_3], [m_3_1, m_3_2, m_3_3]])

M_0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

for n in np.arange(0, l + 1, 1):

	File1 = open("D_L_L_l_{}.dat".format(n), "w")
	File2 = open("D_L_N_l_{}.dat".format(n), "w")
	File3 = open("D_M_M_l_{}.dat".format(n), "w")
	File4 = open("D_N_L_l_{}.dat".format(n), "w")
	File5 = open("D_N_N_l_{}.dat".format(n), "w")

	File6  = open("E_L_L_l_{}.dat".format(n), "w")
	File7  = open("E_L_N_l_{}.dat".format(n), "w")
	File8  = open("E_M_M_l_{}.dat".format(n), "w")
	File9  = open("E_N_L_l_{}.dat".format(n), "w")
	File10 = open("E_N_N_l_{}.dat".format(n), "w")

	File11 = open("F_L_L_l_{}.dat".format(n), "w")
	File12 = open("F_L_N_l_{}.dat".format(n), "w")
	File13 = open("F_M_M_l_{}.dat".format(n), "w")
	File14 = open("F_N_L_l_{}.dat".format(n), "w")
	File15 = open("F_N_N_l_{}.dat".format(n), "w")

	File16  = open("Z_L_L_l_{}.dat".format(n), "w")
	File17  = open("Z_L_N_l_{}.dat".format(n), "w")
	File18  = open("Z_M_M_l_{}.dat".format(n), "w")
	File19  = open("Z_N_L_l_{}.dat".format(n), "w")
	File20  = open("Z_N_N_l_{}.dat".format(n), "w")

	File21  = open("CN_{}.dat".format(n), "w")
	File22  = open("CT_{}.dat".format(n), "w")

	for f in np.arange(1, 2001, 1):

		coef = 2*np.pi*f

		K0L = coef/C0L
		K0T = coef/C0T
		K1L = coef/C1L

		K1T = coef/C1T
		K2L = coef/C2L
		K2T = coef/C2T

######  Matrix's A elements definition #######

		A_1_1 = Matrix_C(n, r2, K0L, K0T, hankel)
		A_1_2 = Matrix_C(n, r2, K2L, K2T, bessel)
		A_1_3 = Matrix_C(n, r2, K2L, K2T, hankel)
	#	A_1_4 = Matrix_0

	#	A_2_1 = Matrix_0
		A_2_2 = Matrix_C(n, r1, K2L, K2T, bessel)
		A_2_3 = Matrix_C(n, r1, K2L, K2T, hankel)
		A_2_4 = Matrix_C(n, r1, K1L, K1T, bessel)

		B_1_1 = Matrix_C(n, r2, K0L, K0T, bessel)
	#	B_2_1 = MAtrix_0
		B_3_1 = Matrix_D(n, r2, K0L, K0T, mu_0, lam_0, bessel)
	#	B_4_1 = MAtrix_0
		
		A_3_1 = Matrix_D(n, r2, K0L, K0T, mu_0, lam_0, hankel)
		A_3_2 = Matrix_D(n, r2, K2L, K2T, mu_2, lam_2, bessel)
		A_3_3 = Matrix_D(n, r2, K2L, K2T, mu_2, lam_2, hankel)
	# 	A_3_4 = MAtrix_0

	#	A_4_1 = MAtrix_0
		A_4_2 = Matrix_D(n, r1, K2L, K2T, mu_2, lam_2, bessel)
		A_4_3 = Matrix_D(n, r1, K2L, K2T, mu_2, lam_2, hankel)
		A_4_4 = Matrix_D(n, r1, K1L, K1T, mu_1, lam_1, bessel)

#############################################################################

		M = np.hstack([A_1_1, -A_1_2, -A_1_3,   M_0 ])
		N = np.hstack([ M_0 ,  A_2_2,  A_2_3, -A_2_4])
		P = np.hstack([A_3_1, -A_3_2, -A_3_3,   M_0 ])
		Q = np.hstack([ M_0 ,  A_4_2,  A_4_3, -A_4_4])

		E = np.vstack([M, N, P, Q])

		F = np.vstack([-B_1_1, M_0, -B_3_1, M_0])

####################################################################################
		M_S   = np.absolute(scipy.linalg.solve(E, F)) # Matrix Solution
		
		File1.write("{:e}\t\t{:e}\n".format(f, M_S[0, 0]))
		File2.write("{:e}\t\t{:e}\n".format(f, M_S[0, 2]))
		File3.write("{:e}\t\t{:e}\n".format(f, M_S[1, 1]))
		File4.write("{:e}\t\t{:e}\n".format(f, M_S[2, 0]))
		File5.write("{:e}\t\t{:e}\n".format(f, M_S[2, 2]))
		File6.write("{:e}\t\t{:e}\n".format(f, M_S[3, 0]))
		File7.write("{:e}\t\t{:e}\n".format(f, M_S[3, 2]))
		File8.write("{:e}\t\t{:e}\n".format(f, M_S[4, 1]))	
		File9.write("{:e}\t\t{:e}\n".format(f, M_S[5, 0]))
		File10.write("{:e}\t\t{:e}\n".format(f, M_S[5, 2]))
		File11.write("{:e}\t\t{:e}\n".format(f, M_S[6, 0]))
		File12.write("{:e}\t\t{:e}\n".format(f, M_S[6, 2]))
		File13.write("{:e}\t\t{:e}\n".format(f, M_S[7, 1]))
		File14.write("{:e}\t\t{:e}\n".format(f, M_S[8, 0]))
		File15.write("{:e}\t\t{:e}\n".format(f, M_S[8, 2]))
		File16.write("{:e}\t\t{:e}\n".format(f, M_S[9, 0]))
		File17.write("{:e}\t\t{:e}\n".format(f, M_S[9, 2]))
		File18.write("{:e}\t\t{:e}\n".format(f, M_S[10, 1]))
		File19.write("{:e}\t\t{:e}\n".format(f, M_S[11, 0]))
		File20.write("{:e}\t\t{:e}\n".format(f, M_S[11, 2]))
		
		CL = M_S[0, 0]**2 + n*(n + 1)*M_S[2, 0]**2
		File21.write("{:e}\t\t{:e}\n".format(f, CL))

		CT = M_S[1, 1]**2 + n*(n + 1)*M_S[0, 2]**2 + M_S[2, 2] 
		File22.write("{:e}\t\t{:e}\n".format(f, CT))

File1.close()
File2.close()
File3.close()
File4.close()
File5.close()
File6.close()
File7.close()
File8.close()
File9.close()
File10.close()
File11.close()
File12.close()
File13.close()
File14.close()
File15.close()
File16.close()
File17.close()
File18.close()
File19.close()
File20.close()
File21.close()
File22.close()
