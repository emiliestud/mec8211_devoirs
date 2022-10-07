"""
Ce code correspond à un code de differences finies pour la question D du devoir 1

"""

import numpy as np
import sympy as sp


#definition des constantes
S = 10**(-8) #mol/m3/s
Deff = 10**(-10) # m2/s
R = 0,5 #m
D = 2*R #m


#definition du nombre de noeud en symbolique pour calcul generique
Ntot = sp.symbols('Ntot')


#definition des grandeurs necessaires au calcul des elements finis
the_C_0 = np.zeros((Ntot,1)) #vecteur des inconnues C a chaque noeud
delta_r =D/(Ntot-1)
delta_t = 0,5 #???????????????????????????????????????????
M = 20 #?????????????????????????????????????????? Ttot = M*delta_t


#ecriture de la matrice de resolution
B = -delta_t*Deff
A = []
the_r = 0.
for i in range (Ntot):
	the_r = the_r + delta_r
	A.append(delta_r**2 + delta_t*Deff*(2 + the_r*delta_r)) 

M = np.zeros((Ntot,Ntot))
M[0,0] = 1.0
M[Ntot,Ntot] = 1
for i in range (1,Ntot-1):
	M[i,i-1] = B
	M[i,i] = A[i]
	m[i,i+1] = B


#definition du terme de droite
D = np.array((Ntot,1))
D[0,0] = Ce
D[Ntot,0] = Ce

for i in range (1, Ntot-1):
	D[i,0] = delta_r**2*the_C_0[i] - S*delta_t*delta_r**2



#calcul du nouveau vecteur de solution a chaque iteration de temps
t = 0
the_C = the_C_0
for k in range (M):
	t+=delta_t
	#update du vecteur D a chaque iteration en temps
	for i in range (1, Ntot-1):
	        D[i,0] = delta_r**2*the_C[i] - S*delta_t*delta_r**2
	#resolution du nouveau vecteur inconnu pour le pas de temps
	the_C = D*np.inverse(M)
print (the_C)

