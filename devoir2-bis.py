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

from xml.etree.ElementInclude import XINCLUDE_FALLBACK
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
k = 4e-9 #s^-1


#### --- definition des grandeurs necessaires au calcul des elements finis --- ###
#vecteur des inconnues C a chaque noeud, initialise avec les conditions initiales
def create_C_0(Ntot):
	the_C_0 = np.zeros((Ntot,1)) 
	the_C_0[-1,0] = Ce
	return the_C_0








####################################################################################
####################################################################################
####################### --- Methode MMS --- ######################
####################################################################################
####################################################################################


"""
Now we don't take S = constant but S = kC instead. in the Finite formulation, we then get for timestep t+1 S = kC(i,t+1).
Then the right vector which depended on the constant C is modified, as well  as the matric M which now includes the term in S = KC(i,t+1)
We do that thanks to the new functions create_R_vector_bis and create_matric_v2_bis. We then need to call a new function solve_C_v2_bis that uses the finite difference
scheme of exercice F of work 1 (as solve_C_v2), but it uses these 2 new functions instead.
The selection of a stationnary or unstationnary equation still works the same, with the boolean stationnary in the main function
"""


#definition du terme de droite avec réaction du premier ordre
def create_R_vector_bis(Ntot,the_C,delta_t,delta_r, stationnary):
	D = np.zeros((Ntot,1))
	D[-1,0] = Ce
	the_r = 0
  
	if stationnary == True:
		alpha = 0.0
	else : 
		alpha = 1.0
    

	for i in range (1,Ntot -1):
		the_r += delta_r
		D[i,0] = alpha*delta_r**2*the_C[i] 
	return D


# POUR LE CAS 2 DE DIFFERENCE FINIE
def create_matrix_v2_bis(Ntot,delta_t,delta_r,stationnary):
	B = -delta_t*Deff
	the_r = 0

	M = np. zeros((Ntot,Ntot)) 
	M[0,0]= -3
	M[0,1] = 4
	M[0,2] = -1
	M[-1,-1] = 1.0 #C5 = Ce = constante

	if stationnary == True:
		alpha = 0.0
	else:
		alpha = 1.0

	for i in range (1,Ntot-1): # de i = 1 a 3
		the_r = the_r + delta_r
		M[i,i-1] = B*(1-delta_r/(2*the_r))
		M[i,i] = alpha*delta_r**2 -B*2 + delta_t*delta_r**2*k
		M[i,i+1] = B*(1 + delta_r/(2*the_r))
	return M


#calcul du nouveau vecteur de solution a chaque iteration de temps
def solve_C_v2_bis(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
	t = 0
	k = 0
	delta_r =R/(Ntot-1)
	the_C = create_C_0(Ntot)
	my_M = create_matrix_v2_bis(Ntot,delta_t,delta_r,stationnary)
	#print(my_M)
	while k < time_iter:
		t+=delta_t
		#update du vecteur D a chaque iteration en temps
		my_D = create_R_vector_bis(Ntot,the_C,delta_t,delta_r,stationnary)
		#resolution du nouveau vecteur inconnu pour le pas de temps
		the_C = np.linalg.solve(my_M,my_D)
		k+=1
	return the_C
#my_C = solve_C_v2_bis(5, 1, 1, False)
#print(my_C)

####################################################################################
# cas 1 : analyse de convergence en espace, travail avec l'équation stationnaire
####################################################################################
# choix et plot de la solution MMS

a = np.pi
k = 4*10**(-9) #s-1
C0 = 10
t0 = 1
t = 1000

def sol_MMS(t,r):
    return C0*np.exp(t/t0)*r**2

radii = np.linspace(0,R,50)
y = []
for r in radii :
    y.append(sol_MMS(t,r))
plt.plot(radii,y)



### Obtention du terme source analytique
radius, time = sp.symbols('radius time')

#from sympy.utilities.lambdify import implemented_function
#terme source MMS
sol_MMS_stationnary = C0*radius**2*sp.exp(time/t0)
#sol_MMS_stationnary = C0*(radius**2)
S_MMS_stationnary = -Deff*(1/radius)*sp.diff(radius*sp.diff(sol_MMS_stationnary,radius),radius) + k*sol_MMS_stationnary
#S_MMS_stationnary = C0*radius*radius*sp.exp(time/t0)*((1/t0)+k)-4*Deff*C0*sp.exp(time/t0)
#S_MMS_stationnary = C0*radius*radius*((1/t0)+k)-4*Deff*C0

#S_MMS_stationnary = C0*((radius**2)/t0 - 4*Deff + k*(radius**2))
#print(S_MMS_stationnary)
#appliquer l'operateur  sur la solution interpolee

#create callable fonction pour l'expression symbolique
f_source_stationnary = sp.lambdify([radius,time], S_MMS_stationnary,"numpy")
f_sol_stationnary = sp.lambdify([radius,time], sol_MMS_stationnary,"numpy")

#visualisation des resultats
#taille du domaine
#xmin = 0.001
#xmax = 0.5
tt=np.zeros((50,1))
tmin = 0
tmax = 1

#set up a regular grid of interpolation points
xdom = np.linspace(0,R,1000)
tdom = np.linspace(tmin,tmax,1000)
xi,ti = np.meshgrid(xdom,tdom)
z_MMS = f_sol_stationnary(xi,ti)
z_source = f_source_stationnary(xi,ti)
z_source_MMS = z_source[0,:]

#evaluate MNP function and source term on the grid
#z_MNP = f_sp(xi)
#z_source_stationnary = []


plt.figure()
plt.contourf(xi,ti,z_source)
plt.colorbar()
plt.xlabel('radius [m]')
plt.ylabel('C [mol/m³]')
plt.title('Terme source MMS')
plt.show()



plt.figure()
plt.contourf(xi,ti,z_MMS)
plt.colorbar()
plt.xlabel('radius [m]')
plt.ylabel('C [mol/m³]')
plt.title('Fonction MMS')
plt.show()


#plt.figure()
#plt.plot(tdom,z_MMS[:,1])
#plt.xlabel('temps [m]')
#plt.ylabel('C [mol/m³]')
#plt.title('solution MNP, cas stationnaire')
#plt.grid()
#plt.show()


#plt.figure()
#plt.plot(xdom,z_MMS[0,:])
#plt.xlabel('radius [m]')
#plt.ylabel('C [mol/m³]')
#plt.title('solution MNP, cas stationnaire')
#plt.grid()
#plt.show()


#plt.plot(xdom,z_source)
#plt.xlabel('radius [m]')
#plt.ylabel('terme source')
#plt.title('Terme source, cas stationnaire')
#plt.grid()
#plt.show()

####################################################################################
# Analyse de convergence en espace et en temps, travail avec l'équation stationnaire puis instationnaire
####################################################################################

#We then need to study the convergence of the code. To do so, we have to add the source term in the right vector.
#definition du terme de droite

def create_R_vector_MMS(Ntot,the_C,delta_t,delta_r, stationnary):
    D = np.zeros((Ntot,1))
    D[-1,0] = Ce
    the_r = 0
    the_t = 0
    #t=1000
    
    if stationnary == True:
        alpha = 0.0
    else : 
        alpha = 1.0
    
    for i in range (1,Ntot-1):
        the_r += delta_r
        the_t += delta_t
        
        source_term = f_source_stationnary(the_r,the_t)
        
        D[i,0] = source_term*delta_t*delta_r**2 + alpha*delta_r**2*the_C[i]
    #print(the_r,t)
    #print(source_term)
    #print(D)
    #print(Ntot)
        
    return D



#calcul du nouveau vecteur de solution a chaque iteration de temps avec inclusion du terme source
def solve_C_v2_MMS(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
    t = 0
    k = 0
    delta_r =R/(Ntot-1)
    the_C = create_C_0(Ntot)
    my_M = create_matrix_v2_bis(Ntot,delta_t,delta_r,stationnary)
	#print(my_M)
    while k <= time_iter:
        t+=delta_t
    		#update du vecteur D a chaque iteration en temps
        my_D = create_R_vector_MMS(Ntot,the_C,delta_t,delta_r,stationnary)
    		#resolution du nouveau vecteur inconnu pour le pas de temps
        the_C = np.linalg.solve(my_M,my_D)
        k+=1
    #print("len :",len(the_C))
    #print(my_M)
    return the_C
my_C_MMS = solve_C_v2_MMS(1000,0.001,1,False)
#print(my_C_MMS)
#print("len final :",len(my_C_MMS))
#print(my_C_MMS)



#######################################
### ---- Calcul de l'erreur L1 ---- ###
#######################################

def erreur_L1_MMS(u_num,u_anal,Ntot):
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

def erreur_L2_MMS(u_num,u_anal,Ntot):
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


def erreur_inf_MMS(u_num,u_anal,Ntot):
	err = 0.0
	for n in range (Ntot):
		if abs(u_num[n]-u_anal[n]) >= err:
			err = abs(u_num[n]-u_anal[n])
	return err

list_ntot = [5,10,15,20,40,60,100]
erreurs_1 = []
erreurs_2 = []
erreurs_3 = []
Ntot=100
Nmax=1000
u_anal = solve_C_v2_MMS(Nmax,0.001,1,False)
radii2 = np.linspace(0,R,Nmax)

list_u_anal = z_MMS[0,:]
radii = np.linspace(0,R,Ntot)

for Ntot in list_ntot:
	radii = np.linspace(0,R,Ntot)	
	my_C = solve_C_v2_MMS(Ntot,0.001,1,False)
	err1 = erreur_L1_MMS(my_C,list_u_anal,Ntot)
	err2 = erreur_L2_MMS(my_C,list_u_anal,Ntot)
	err3 = erreur_inf_MMS(my_C,list_u_anal,Ntot)
	erreurs_1.append(err1)
	erreurs_2.append(err2)
	erreurs_3.append(err3)



list_h = []
deltax_ref = R/Nmax
for Ntot in list_ntot:
	deltax = R/Ntot
	list_h.append(deltax/deltax_ref)


plt.figure()
plt.loglog(list_h,erreurs_1, label = 'erreur L1',marker = '.')
plt.loglog(list_h,erreurs_2, label = 'erreur L2',marker = '.')
plt.loglog(list_h,erreurs_3, label = 'erreur infinie',marker = '.')
plt.legend()
plt.grid()
plt.title('erreurs dans le cas stationnaire')
plt.xlabel('epaisseur de maillage h [-]')
plt.ylabel('erreurs [-]')
plt.title("erreurs vs $h = \Delta x / \Delta x_{ref}$, equation stationnaire")
plt.show()


























radii = np.linspace(0,R,1000)
#print(my_C)
plt.figure()
plt.plot(radii,my_C_MMS,label = 'numerique')
plt.plot(radii,z_MMS[0,:],label = 'fonction source',linestyle = 'none', marker = '.')
plt.legend()
#print(stationnary_analytic_C(5))


#######################################
### ---- Calcul de l'erreur L1 ---- ###
#######################################

def erreur_L1_MMS(u_num,u_anal,Ntot):
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

def erreur_L2_MMS(u_num,u_anal,Ntot):
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


def erreur_inf_MMS(u_num,u_anal,Ntot):
	err = 0.0
	for n in range (Ntot):
		if abs(u_num[n]-u_anal[n]) >= err:
			err = abs(u_num[n]-u_anal[n])
	return err

###########################################
###########################################
###########################################
list_ntot = [5,10,20,40,80,1000]
#list_ntot = [5]
erreurs_1 = []
erreurs_2 = []
erreurs_3 = []

for Ntot in list_ntot:
    tmin = 0
    tmax = 1

#set up a regular grid of interpolation points
    xdom = np.linspace(0,R,Ntot)
    tdom = np.linspace(tmin,tmax,Ntot)
    xi,ti = np.meshgrid(xdom,tdom)
    z_MMS = f_sol_stationnary(xi,ti)
    Z_MMS_RADIUS = z_MMS[0,:]
    z_source = f_source_stationnary(xi,ti)
    the_radii = np.linspace(0,R,Ntot)
    my_C = solve_C_v2_MMS(Ntot,0.01,100,True)
    u_anal = z_MMS[0,:]
    err1 = erreur_L1_MMS(my_C,u_anal,Ntot)
    err2 = erreur_L2_MMS(my_C,u_anal,Ntot)
	#err3 = erreur_inf_MMS(my_C,u_anal,Ntot)
    erreurs_1.append(err1)
    erreurs_2.append(err2)
	#erreurs_3.append(err3)


plt.figure()
plt.loglog(list_ntot,erreurs_1, label = 'erreur L1')
plt.loglog(list_ntot,erreurs_2, label = 'erreur L2')
#plt.loglog(list_ntot,erreurs_3, label = 'erreur infinie')
plt.legend()
plt.title('erreurs dans le cas stationnaire')
plt.xlabel('log du nombre de noeuds')
plt.ylabel('log(erreur)')
plt.title('erreurs')
#plt.show()





