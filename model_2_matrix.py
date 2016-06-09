#########  Matrix Solve  ##############

import numpy as np
import scipy
import argparse
from scipy import linalg
from scipy import special
np.set_printoptions(threshold=np.nan)

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
##
M_0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

a = np.zeros([16, 2001]) # Matrices de 2000 filas por 16 columnas inicializadas con elementos cero
b = np.zeros([16, 2001])
	
for f in np.arange(1, 2001, 1):
	#print f
	ColSig_L = [] # Lista de elementos vacio que contendra los valores de sigma
	ColSig_T = []
	
	for n in np.arange(0, l + 1, 1):

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
####
		M = np.hstack([A_1_1, -A_1_2, -A_1_3,   M_0 ])
		N = np.hstack([ M_0 ,  A_2_2,  A_2_3, -A_2_4])
		P = np.hstack([A_3_1, -A_3_2, -A_3_3,   M_0 ])
		Q = np.hstack([ M_0 ,  A_4_2,  A_4_3, -A_4_4])
		E = np.vstack([M, N, P, Q])
		F = np.vstack([-B_1_1, M_0, -B_3_1, M_0])
####
		M_S = np.absolute(scipy.linalg.solve(E, F)) # Matrix Solution

		sigma_L = (4*(2*n + 1)/(K0L*r2)**2)*(M_S[0, 0]**2 + n*(n + 1)*((C0T/C0L)**3)*M_S[2, 0])
		
		n = n + 1
		sigma_T = (2*(2*n + 1)/(K0T*r2)**2)*((1/(n*(n + 1)))*((C0T/C0L)**3)*M_S[0, 2]**2 + M_S[1, 1]**2 + M_S[2, 2])

		#print sigma_L
		
		ColSig_L.append(sigma_L)
		ColSig_T.append(sigma_T)
	
	#print ColSig_L
	
	a[:, f] = ColSig_L
	b[:, f] = ColSig_T

	MatSigmaL = a
	MatSigmaT = b

np.savetxt('sigma_L.dat', MatSigmaL.sum(axis=0), fmt='%.5e', delimiter=' ')
np.savetxt('sigma_T.dat', MatSigmaT.sum(axis=0), fmt='%.5e', delimiter=' ')