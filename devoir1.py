"""
Ce code correspond a un code de differences finies pour la question D du devoir 1
Ce code est bien generique par rapport a Ntot car ici Ntot est un parametre utilise par un ensemble de fonctions.
La fonction principale de resolution est solve_C, qui utilise d'auters fonctions qui servent a definir les matrices de difference finies

"""

import numpy as np
import sympy as sp
from numpy.linalg import inv


#definition des constantes
S = 10**(-8) #mol/m3/s
Deff = 10**(-10) # m2/s
R = 0.5 #m
D = 2*R #m
Ce = 10 #mol/m3

#definition des grandeurs necessaires au calcul des elements finis
def create_C_0(Ntot):
	the_C_0 = np.zeros((Ntot,1)) #vecteur des inconnues C a chaque noeud
	the_C_0[0,0] = Ce
	the_C_0[-1,0] = Ce
	return the_C_0

print(create_C_0(5))


#ecriture de la matrice de resolution
def create_matrix(Ntot,delta_t,delta_r):
	B = -delta_t*Deff
	A = []
	the_r = 0.
	for i in range (Ntot):
		the_r = the_r + delta_r
		#A.append(delta_t*Deff*(2 + 1/the_r*delta_r)) pour cas stationnaire
		A.append(delta_r**2 + delta_t*Deff*(2 + 1/the_r*delta_r)) 

	M = np.zeros((Ntot,Ntot))
	M[0,0] = 1.0
	M[-1,-1] = 1
	for i in range (1,Ntot-1):
		M[i,i-1] = B
		M[i,i] = A[i]
		M[i,i+1] = B
	return M


#definition du terme de droite
def create_R_vector(Ntot,the_C,delta_t,delta_r):
	D = np.zeros((Ntot,1))
	D[0,0] = Ce
	D[-1,0] = Ce
	for i in range (1, Ntot-1):
		#D[i,0] = - S*delta_t*delta_r**2 pour cas stationnaire
		D[i,0] = delta_r**2*the_C[i] - S*delta_t*delta_r**2
	return D


#calcul du nouveau vecteur de solution a chaque iteration de temps
def solve_C(Ntot,delta_t,time_iter):
	t = 0
	k = 0
	delta_r =D/(Ntot-1)
	the_C = create_C_0(Ntot)
	my_M = create_matrix(Ntot,delta_t,delta_r)
	#print(my_M)
	while k < time_iter:
		t+=delta_t
		#update du vecteur D a chaque iteration en temps
		my_D = create_R_vector(Ntot,the_C,delta_t,delta_r)
		#print(my_D)
		#resolution du nouveau vecteur inconnu pour le pas de temps
		my_M_inv = np.linalg.inv(my_M)
		#print(my_M_inv)
		the_C = np.matmul(my_M_inv,my_D)
		#the_C = np.linalg.solve(my_M,my_D)
		k+=1
	return the_C

my_C = solve_C(5,1,100)
print(my_C)


def stationnary_analytic_C(Ntot):
	anal_C = np.zeros((Ntot,1)) #for comparisons
	the_r = np.array([-R,-R/2,0,R/2,R])
	for i in range (Ntot):
		r = the_r[i]
		anal_C[i] = 1/4*S/Deff*R**2*((r/R)**2-1)+Ce
	return anal_C

print(stationnary_analytic_C(5))

#cas stationnaire : pas de terme en (i-1,t+1) ni en (i,t-1)