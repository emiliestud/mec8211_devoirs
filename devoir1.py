"""
Ce code correspond a un code de differences finies pour la question D du devoir 1
Ce code est bien generique par rapport a Ntot car ici Ntot est un parametre utilise par un ensemble de fonctions.
La fonction principale de resolution est solve_C, qui utilise d'auters fonctions qui servent a definir les matrices de difference finies

Pour lancer une solution, il faut utilise la fonction solve_C_v1, ou solve_C_v2 en fonction du schéma de différences finies souhaité. 
Cette fonction utilise les autres fonctions antérieures de création de matrices pour résoudre le probleme.
Les parametres de ces fonction ssont : 
- le nombre de noeuds total Ntot
- le pas de temps delta_t
- le nombre d'iterations en temps desirees time_iter
- si on choisit d'utiliser le schema de difference finies pour lecas stationnaire ou non bool:stationnary

"""

import numpy as np
import sympy as sp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy as scp
from scipy import interpolate


#### --- definition des constantes --- ###
S = 10**(-8) #mol/m3/s
Deff = 10**(-10) # m2/s
R = 0.5 #m
D = 2*R #m
Ce = 10 #mol/m3


#### --- definition des grandeurs necessaires au calcul des elements finis --- ###
#vecteur des inconnues C a chaque noeud, initialise avec les conditions initiales
def create_C_0(Ntot):
	the_C_0 = np.zeros((Ntot,1)) 
	the_C_0[-1,0] = Ce
	return the_C_0


#ecriture de la matrice de resolution 
# POUR LE CAS 1 DE DIFFERENCE FINIE
def create_matrix_v1(Ntot,delta_t,delta_r, stationnary):
	B = -delta_t*Deff
	the_r = 0

	M = np.zeros((Ntot,Ntot)) 
	M[0,1] = 1.0 # C0 = C1
	M[0,0] = -1.0
	M[-1,-1] = 1.0 #C5 = Ce = constante
	if stationnary == True :
		alpha = 0.0
	else:
		alpha = 1.0
	for i in range (1,Ntot-1): # de i = 1 a 3
		the_r = the_r + delta_r
		M[i,i-1] = B
		M[i,i] = alpha*delta_r**2 -B*(2 + delta_r/the_r)
		M[i,i+1] = B*(1 + delta_r/the_r)
	return M


#definition du terme de droite
def create_R_vector(Ntot,the_C,delta_t,delta_r, stationnary):
	D = np.zeros((Ntot,1))
	D[-1,0] = Ce
	the_r = 0

	if stationnary == True:
		alpha = 0.0
	else : 
		alpha = 1.0

	for i in range (1,Ntot -1):
		the_r += delta_r
		D[i,0] = - S*delta_t*delta_r**2 + alpha*delta_r**2*the_C[i]
	return D


#calcul du nouveau vecteur de solution a chaque iteration de temps
def solve_C_v1(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
	t = 0
	k = 0
	delta_r =R/(Ntot-1)
	the_C = create_C_0(Ntot)
	my_M = create_matrix_v1(Ntot,delta_t,delta_r,stationnary)
	#print(my_M)
	while k < time_iter:
		t+=delta_t
		#update du vecteur D a chaque iteration en temps
		my_D = create_R_vector(Ntot,the_C,delta_t,delta_r,stationnary)
		#resolution du nouveau vecteur inconnu pour le pas de temps
		the_C = np.linalg.solve(my_M,my_D)
		k+=1
	return the_C

#my_C = solve_C_v1(500,1,1000,False)
#print(my_C)






# POUR LE CAS 2 DE DIFFERENCE FINIE
def create_matrix_v2(Ntot,delta_t,delta_r,stationnary):
	B = -delta_t*Deff
	the_r = 0

	M = np.zeros((Ntot,Ntot)) 
	M[0,1] = 4/3
	M[0,2] = -1/3
	M[-1,-1] = 1.0 #C5 = Ce = constante

	if stationnary == True:
		alpha = 0.0
	else:
		alpha = 1.0

	for i in range (1,Ntot-1): # de i = 1 a 3
		the_r = the_r + delta_r
		M[i,i-1] = B*(1-delta_r/(2*the_r))
		M[i,i] = alpha*delta_r**2 -B*2
		M[i,i+1] = B*(1 + delta_r/(2*the_r))
	return M


#calcul du nouveau vecteur de solution a chaque iteration de temps
def solve_C_v2(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
	t = 0
	k = 0
	delta_r =R/(Ntot-1)
	the_C = create_C_0(Ntot)
	my_M = create_matrix_v2(Ntot,delta_t,delta_r,stationnary)
	#print(my_M)
	while k < time_iter:
		t+=delta_t
		#update du vecteur D a chaque iteration en temps
		my_D = create_R_vector(Ntot,the_C,delta_t,delta_r,stationnary)
		#resolution du nouveau vecteur inconnu pour le pas de temps
		the_C = np.linalg.solve(my_M,my_D)
		k+=1
	return the_C

my_C = solve_C_v2(5,1,1,True)
#print(my_C)





##################################################
### ---- Solution analytique stationnaire ---- ###
##################################################

def stationnary_analytic_C(Ntot):
	anal_C = np.zeros((Ntot,1)) #for comparisons
	the_r = np.linspace(0,R,Ntot)
	for i in range (Ntot):
		r = the_r[i]
		anal_C[i] = 1/4*S/Deff*R**2*((r/R)**2-1)+Ce
	return anal_C


my_C = solve_C_v2(5,1,1,True)
radii = [0.,R/4,R/2,3*R/4,R]
#print(my_C)
plt.figure()
plt.plot(radii,my_C,label = 'numerique')
plt.plot(radii,stationnary_analytic_C(5),label = 'analytique')
plt.legend()
#print(stationnary_analytic_C(5))

#cas stationnaire : pas de terme en (i-1,t+1) ni en (i,t-1), et pas de temps delta_t = 1

#######################################
### ---- Calcul de l'erreur L1 ---- ###
#######################################

def erreur_L1(u_num,u_anal,Ntot):
	err = 0.0
	r = 0
	delta_r = D/(Ntot-1)
	for n in range (Ntot):
		r_n = r + delta_r
		err += r_n+abs(u_num[n]-u_anal[n])
	err = err/Ntot
	return err

#######################################
### ---- Calcul de l'erreur L2 ---- ###
#######################################

def erreur_L2(u_num,u_anal,Ntot):
	err = 0.0
	r = 0
	delta_r = D/(Ntot-1)
	for n in range (Ntot):
		r_n = r + delta_r
		err += r_n+abs(u_num[n]-u_anal[n])**2
	err = err/Ntot
	err = np.sqrt(err)
	return err


############################################
### ---- Calcul de l'erreur inifnie ---- ###
############################################


def erreur_inf(u_num,u_anal,Ntot):
	err = 0.0
	for n in range (Ntot):
		if abs(u_num[n]-u_anal[n]) >= err:
			err = abs(u_num[n]-u_anal[n])
	return err

###########################################
###########################################
###########################################
#list_ntot = [5,10,15,20,40,60]
list_ntot = [5]
erreurs_1 = []
erreurs_2 = []
erreurs_3 = []

for Ntot in list_ntot:
	the_radii = np.linspace(0,R,Ntot)
	my_C = solve_C_v1(Ntot,1,50,False)
	u_anal = stationnary_analytic_C(Ntot)
	err1 = erreur_L1(my_C,u_anal,Ntot)
	err2 = erreur_L2(my_C,u_anal,Ntot)
	err3 = erreur_inf(my_C,u_anal,Ntot)
	erreurs_1.append(err1)
	erreurs_2.append(err2)
	erreurs_3.append(err3)


plt.figure()
plt.loglog(list_ntot,erreurs_1, label = 'erreur L1')
#plt.loglog(list_ntot,erreurs_2, label = 'erreur L2')
#plt.loglog(list_ntot,erreurs_3, label = 'erreur infinie')
plt.legend()
plt.title('erreurs dans le cas stationnaire')
plt.xlabel('log du nombre de noeuds')
plt.ylabel('log(erreur)')
#plt.show()


###################################################################################
####################### --- Methode des problemes proches ---######################
###################################################################################


### Calcul d'un solution avec un maillage tres fin avec Ntot = 60
Ntot = 60
C_analytical = solve_C_v1(Ntot,1,1,True)


### Interpolation Analytique
the_r = np.linspace(0,R,Ntot)
#print(the_r)
#print(len( C_analytical[:,0]))
f = scp.interpolate.interp1d(the_r, C_analytical[:,0],'cubic')
the_r_new = the_r
C_interpolated = f(the_r_new)
#print(np.size(C_interpolated))
plt.figure()
plt.plot(the_r_new,C_interpolated)
plt.show()

### Obtention du terme source analytique

radius = sp.symbols('radius')
#Solution interpolee est derivee'
f_sp = sp.Function('f')
C_interpolated = f_sp(radius)

#appliquer l'operateur  sur la solution interpolee
source_stationnary = Deff*1/radius*sp.diff(radius*sp.diff(C_interpolated,radius),radius)-S

#create callable fonction pour l'expression symbolique
f_source = sp.lambdify(radius,source_stationnary)

my_source = f_source(the_r_new)
plt.figure()
plt.plot(the_r_new,my_source)
plt.title('terme source')
plt.show()