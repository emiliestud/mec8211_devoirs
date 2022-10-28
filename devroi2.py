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


####################################################################################
####################################################################################
####################### --- Methode des problemes proches --- ######################
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

####################################################################################
# cas 1 : analyse de convergence en espace, travail avec l'équation instationnaire
####################################################################################

### Calcul d'un solution avec un maillage tres fin avec Ntot = 60
Ntot = 100
delta_t = 10
time_iter = 10
Ttot=delta_t*time_iter
C_analyticals_1 = []
for my_time_iter in range(time_iter):
    C_analytical_1 = solve_C_v2_bis(Ntot,delta_t,my_time_iter,False)
    C_analyticals_1.append(C_analytical_1)
print(np.shape(C_analyticals_1))
radii = np.linspace(0,R,Ntot)
times = np.arange(0,Ttot,delta_t)

#Interpolation de la solution analytique
RR,TT= np.meshgrid(radii,times)
C_1 = scp.interpolate.interp2d(radii,times, C_analyticals_1,'cubic')



### Obtention du terme source analytique
radius,time = sp.symbols('radius time')

from sympy.utilities.lambdify import implemented_function
#Solution MNP interpolee est derivee'
f_1_MNP = implemented_function('f_1_MNP',C_1)
# C_sp_stationnary = sp.Function('C_stationnary')
#C_sp_stationnary = sp.symbols('f_sp_stationnary', cls = function) 
#C_interpolated = f_sp(radius)

#appliquer l'operateur  sur la solution interpolee
source_1 = sp.diff(f_1_MNP,time)-Deff*1/radius*sp.diff(radius*sp.diff(f_1_MNP,radius),radius)+k
#create callable fonction pour l'expression symbolique
f_source_1 = sp.lambdify([time,radius],source_1,"numpy")

#visualisation des resultats
#taille du domaine
rmin = 0
rmax = R
tmin = 0
tmax = delta_t*time_iter

rdom = np.linspace(rmin,rmax,1000)
tdom = np.linspace(tmin,tmax,1000)
ri,ti = np.meshgrid(rdom,tdom)

#set up a regular grid of interpolation points
#z_MNP = C_1(radii,times)
z_MNP_1 = C_1(rdom,tdom)
z_source_1 = f_source_1(ri,ti)

# plt.contour(ri,ti,z_MNP_1)
# plt.colorbar()
# plt.title('Fonction analytique MNP [$mol/m^3$]')
# plt.xlabel('radius [m]')
# plt.ylabel('time [s]')
# plt.show()

y = []
y_source = []
for r in radii:
    y.append(C_1(r,times[-1]))
    y_source.append(f_source_1(r,times[-1]))

plt.figure()
plt.plot(radii,y)
plt.xlabel('radius [m]')
plt.grid()
plt.ylabel('C [mol/m³]')
plt.title('solution MNP à $t = t_f$')
plt.show()

plt.figure()
plt.plot(radii,y_source)
plt.xlabel('radius [m]')
plt.grid()
plt.ylabel('C [mol/m³]')
plt.title('solution MNP à $t = t_f$')
plt.show()


plt.contour(ri,ti,z_source_1)
plt.colorbar()
plt.title('Fonction analytique MNP [$mol/m^3$]')
plt.xlabel('radius [m]')
plt.ylabel('time [s]')
plt.show()
# plt.figure()

# plt.plot(radii,z_source_1,label = 'instationnaire')
# plt.xlabel('radius [m]')
# plt.ylabel('terme source [mol/m³]')
# plt.legend()
# plt.title('Terme source, cas instationnaire')
# plt.grid()
# plt.show()


# ####################################################################################
# # cas 2 : analyse de convergence en temps, travail avec l'équation instationnaire
# ####################################################################################

### Calcul d'un solution avec un maillage tres fin avec Ntot = 60
Ntot = 100
Ttot = 10
delta_t = 10
time_iter = Ttot/delta_t
radii_2 = np.linspace(0,R,Ntot)

C_analyticals_2 = C_analyticals_1
# for my_time_iter in range(time_iter):
#     C_analytical_2 = solve_C_v2_bis(Ntot,delta_t,my_time_iter,False)
#     C_analyticals_2.append(C_analytical_2)
# print(np.shape(C_analyticals_1))
# radii = np.linspace(0,R,Ntot)
# times = np.arange(0,Ttot,delta_t)

#Interpolation de la solution analytique
RR,TT= np.meshgrid(radii_2,times)
C_2 = scp.interpolate.interp2d(radii_2,times, C_analyticals_2,'cubic')
f_2_MNP = implemented_function('f_1_MNP',C_2)

# #Interpolation de la solution analytique
# C_2 = scp.interpolate.interp1d(radii, C_analytical_2[:,0],'cubic')


# plt.figure()
# plt.plot(radii,C_analytical_2, label = 'instationnaire')
# plt.title('solutions interpolées')
# plt.xlabel('radius [m]')
# plt.legend()
# plt.ylabel('C [mol/m³]')
# plt.grid()
# plt.show()


# from sympy.utilities.lambdify import implemented_function
# #Solution MNP interpolee est derivee'
# f_2_MNP = implemented_function('f_stationnary_MNP',C_stationnary)
# # C_sp_stationnary = sp.Function('C_stationnary')
# #C_sp_stationnary = sp.symbols('f_sp_stationnary', cls = function) 
# #C_interpolated = f_sp(radius)

#appliquer l'operateur  sur la solution interpolee
#source_2 = sp.diff(f_2_MNP,time)-Deff*1/radius*sp.diff(radius*sp.diff(f_2_MNP,radius),radius)+k*f_2_MNP.evalf(time,radius)
source_2 = sp.diff(f_2_MNP,time)-Deff*1/radius*sp.diff(radius*sp.diff(f_2_MNP,radius),radius)+k
#create callable fonction pour l'expression symbolique
f_source_2 = sp.lambdify([radius,time],source_2,"numpy")

#visualisation des resultats
#taille du domaine
# rmin = 0
# rmax = R
# tmin = 0
# tmax = delta_t*time_iter

# rdom = np.linspace(rmin,rmax,1000)
# tdom = np.linspace(tmin,tmax,1000)
# ri,ti = np.meshgrid(rdom,tdom)

# #set up a regular grid of interpolation points
# #z_MNP = C_1(radii,times)
# z_MNP_2 = C_1(rdom,tdom)
# z_source_2 = f_source_1(ri,ti)


# yy = []
# yy_source = []
# for r in radii:
#     my_yy = []
#     my_yy_source = []
#     for t in times:
#         my_yy.append(C_2(r,t))
#         my_yy_source.append(f_source_2(r,t))
#     yy.append(my_yy)
#     yy_source.append(my_yy_source)



# plt.figure()
# for k in range(len(radii)):
#     plt.plot(times,yy[k],label = 'r = %f ' %radii[k])
# plt.legend()
# plt.xlabel('t [s]')
# plt.grid()
# plt.ylabel('C [mol/m³]')
# plt.title('solution MNP')
# plt.show()

# plt.figure()
# plt.plot(radii,y_source)
# plt.xlabel('radius [m]')
# plt.grid()
# plt.ylabel('C [mol/m³]')
# plt.title('solution MNP à $t = t_f$')
# plt.show()

# plt.figure()
# plt.plot(radii_2,y)
# plt.xlabel('radius [m]')
# plt.grid()
# plt.ylabel('C [mol/m³]')
# plt.title('solution MNP à $t = t_f$')
# plt.show()

# plt.figure()
# plt.plot(radii,z_source_1,label = 'cas 1')
# plt.plot(radii_2,z_source_2,label = 'cas 2')
# plt.xlabel('radius [m]')
# plt.ylabel('terme source [mol/m³]')
# plt.legend()
# plt.title('Terme source, cas instationnaire')
# plt.grid()
# plt.show()


# ###################################################################################
# ##Analyse de convergence en espace et en temps, travail avec l'équation stationnaire puis instationnaire
# ###################################################################################

# ##We then need to study the convergence of the code. To do so, we have to add the source term in the right vector.
##definition du terme de droite

def create_R_vector_ter(Ntot,the_C,delta_t,delta_r, stationnary):
    D = np.zeros((Ntot,1))
    D[-1,0] = Ce
    the_t = 0
    the_r = 0
    if stationnary == True:
        alpha = 0.0
    else : 
        alpha = 1.0

    for i in range (1,Ntot -1):
        the_r += delta_r
        the_t += delta_t
        source_term = f_source_2(the_r,the_t)
        D[i,0] = source_term*delta_t*delta_r**2 + alpha*delta_r**2*the_C[i]
        #print(source_term)
    return D

#calcul du nouveau vecteur de solution a chaque iteration de temps avec inclusion du terme source
def solve_C_v2_ter(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
	t = 0
	k = 0
	delta_r =R/(Ntot-1)
	the_C = create_C_0(Ntot)
	my_M = create_matrix_v2_bis(Ntot,delta_t,delta_r,stationnary)
	#print(my_M)
	while k < time_iter:
		t+=delta_t
		#update du vecteur D a chaque iteration en temps
		my_D = create_R_vector_ter(Ntot,the_C,delta_t,delta_r,stationnary)
		#resolution du nouveau vecteur inconnu pour le pas de temps
		the_C = np.linalg.solve(my_M,my_D)
		k+=1
	return the_C

# ###################################
# #cas stationnaire, erreur en espace

#MNP_final_C_stationnary = solve_C_v2_ter(Ntot,1,1,True)
MNP_final_C_unstationnary = solve_C_v2_ter(Ntot,delta_t,time_iter,False)


list_ntot = [5,10,15,20,40,60,100]
erreurs_1 = []
erreurs_2 = []
erreurs_3 = []

Nmax=1000
u_anal = solve_C_v2_ter(Nmax,1,1,True)
radii2 = np.linspace(0,R,Nmax)
f_u_anal = scp.interpolate.interp1d(radii2, u_anal[:,0],'cubic')
radii = np.linspace(0,R,Ntot)

for Ntot in list_ntot:
	radii = np.linspace(0,R,Ntot)
	list_u_anal = []
	for r in radii:
		list_u_anal.append(f_u_anal(r))
	the_radii = np.linspace(0,R,Ntot)
	my_C = solve_C_v2_ter(Ntot,1,1,True)
	err1 = erreur_L1(my_C,list_u_anal,Ntot)
	err2 = erreur_L2(my_C,list_u_anal,Ntot)
	err3 = erreur_inf(my_C,list_u_anal,Ntot)
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



# ###################################
# #cas unstationnaire, erreur en temps

# # list_dts = [0.1,0.2,0.4,.8,1.0,2.0,10.]
# # erreurs_1 = []
# # erreurs_2 = []
# # erreurs_3 = []

# # Ntot = 500
# # delta_t_min =1
# # time_iter_min = Ttot/delta_t_min
# # u_anal = solve_C_v2_ter(Ntot,delta_t_min,time_iter,True)
# # radii2 = np.linspace(0,R,Ntot)
# # radii = np.linspace(0,R,Ntot)

# # for delta_t in list_dts:
# # 	time_iter = Ttot/delta_t
# # 	my_C = solve_C_v2_ter(Ntot,delta_t,time_iter,False)
# # 	err1 = erreur_L1(my_C,u_anal,Ntot)
# # 	err2 = erreur_L2(my_C,u_anal,Ntot)
# # 	err3 = erreur_inf(my_C,u_anal,Ntot)
# # 	erreurs_1.append(err1)
# # 	erreurs_2.append(err2)
# # 	erreurs_3.append(err3)



# plt.figure()
# plt.loglog(list_dts,erreurs_1, label = 'erreur L1',marker = '.')
# plt.loglog(list_dts,erreurs_2, label = 'erreur L2',marker = '.')
# plt.loglog(list_dts,erreurs_3, label = 'erreur infinie',marker = '.')
# plt.legend()
# plt.grid()
# plt.title('erreurs dans le cas instationnaire')
# plt.xlabel('epaisseur de maillage h [-]')
# plt.ylabel('erreurs [-]')
# plt.title("erreurs vs $\Delta t$, equation instationnaire")
# plt.show()



# ####################################################################################
# ####################################################################################
# ####################### --- Methode des problemes proches --- ######################
# ####################################################################################
# ####################################################################################


# ####################################################################################
# # cas 1 : analyse de convergence en espace, travail avec l'équation stationnaire
# ####################################################################################

# ### Obtention du terme source analytique
# radius,time = sp.symbols('radius time')
# C0 = Ce
# T0 = 1
# C_analytical = C0*radius*radius*sp.exp(time/T0)
# f_C_analytical = sp.lambdify([radius,time],C_analytical)

# the_times = np.zeros((10,1))
# plt.figure()
# plt.plot(radii,f_C_analytical(radii,the_times[0]))
# plt.title('MMS - solution analytique stationnaire interpolee')
# plt.xlabel('radius [m]')
# plt.ylabel('C [mol/m³]')
# plt.grid()
# plt.show()



# #appliquer l'operateur  sur la solution interpolee
# MMS_source_stationnary = -Deff*1/radius*sp.diff(radius*sp.diff(C_analytical,radius),radius)+k*C_analytical
# #create callable fonction pour l'expression symbolique
# f_MMS_source_stationnary = sp.lambdify([radius,time],MMS_source_stationnary,"numpy")

# #visualisation des resultats
# #taille du domaine
# xmin = 0.001
# xmax = 0.5
# tmin = 0
# tmax = 10

# #set up a regular grid of interpolation points
# xdom = np.linspace(0,R,50)
# tdom = np.linspace(tmin,tmax,50)
# xi,ti = np.meshgrid(xdom,tdom)
# #z_MNP = C_stationnary(xi,ti)
# z_source_MMS = f_MMS_source_stationnary(xi,ti)


# plt.contour(xi,ti,z_source_MMS)
# plt.colorbar()
# plt.xlabel('radius [m]')
# plt.ylabel('C [mol/m³]')
# plt.title('solution MNP, cas stationnaire')
# plt.grid()
# plt.show()


# plt.figure()
# plt.plot(radii,z_source_MMS)
# plt.xlabel('radius [m]')
# plt.ylabel('terme source [mol/m³]')
# plt.title('Terme source, cas stationnaire')
# plt.grid()
# plt.show()


# ####################################################################################
# # cas 2 : analyse de convergence en temps, travail avec l'équation instationnaire
# ####################################################################################


# #appliquer l'operateur  sur la solution interpolee
# MMS_source_unstationnary = sp.diff(C_analytical,time)-Deff*1/radius*sp.diff(radius*sp.diff(C_analytical,radius),radius)+k*C_analytical
# #create callable fonction pour l'expression symbolique
# f_MMS_source_unstationnary = sp.lambdify(radius,MMS_source_unstationnary,"numpy")

# #visualisation des resultats
# #taille du domaine
# xmin = 0.001
# xmax = 0.5

# #set up a regular grid of interpolation points
# xdom = np.linspace(0,R,50)
# xi = np.meshgrid(xdom)
# z_MNP = C_unstationnary(xi)
# z_source_unstationnary = f_MMS_source_unstationnary(radii)



# plt.figure()
# plt.plot(radii,z_source,label = 'stationnaire')
# plt.plot(radii,z_source_unstationnary,label = 'instationnaire')
# plt.xlabel('radius [m]')
# plt.ylabel('terme source [mol/m³]')
# plt.legend()
# plt.title('Terme source, cas instationnaire')
# plt.grid()
# plt.show()


# ####################################################################################
# # Analyse de convergence en espace et en temps, travail avec l'équation stationnaire puis instationnaire
# ####################################################################################

# #We then need to study the convergence of the code. To do so, we have to add the source term in the right vector.
# #definition du terme de droite

# def create_R_vector_ter(Ntot,the_C,delta_t,delta_r, stationnary):
# 	D = np.zeros((Ntot,1))
# 	D[-1,0] = Ce
# 	the_r = 0

# 	if stationnary == True:
# 		alpha = 0.0
# 	else : 
# 		alpha = 1.0

# 	for i in range (1,Ntot -1):
# 		the_r += delta_r
# 		if stationnary == True :
# 			source_term = f_source_stationnary(the_r)
# 		else :
# 			source_term = f_source_unstationnary(the_r)
# 		D[i,0] = source_term*delta_t*delta_r**2 + alpha*delta_r**2*the_C[i]
# 	return D

# #calcul du nouveau vecteur de solution a chaque iteration de temps avec inclusion du terme source
# def solve_C_v2_ter(Ntot,delta_t,time_iter,stationnary): #stationnary = Bool
# 	t = 0
# 	k = 0
# 	delta_r =R/(Ntot-1)
# 	the_C = create_C_0(Ntot)
# 	my_M = create_matrix_v2_bis(Ntot,delta_t,delta_r,stationnary)
# 	#print(my_M)
# 	while k < time_iter:
# 		t+=delta_t
# 		#update du vecteur D a chaque iteration en temps
# 		my_D = create_R_vector_ter(Ntot,the_C,delta_t,delta_r,stationnary)
# 		#resolution du nouveau vecteur inconnu pour le pas de temps
# 		the_C = np.linalg.solve(my_M,my_D)
# 		k+=1
# 	return the_C

# ###################################
# #cas stationnaire, erreur en espace

# #MNP_final_C_stationnary = solve_C_v2_ter(Ntot,1,1,True)
# #MNP_final_C_unstationnary = solve_C_v2_ter(Ntot,delta_t,time_iter,False)


# list_ntot = [5,10,15,20,40,60]
# #list_ntot = [5]
# erreurs_1 = []
# erreurs_2 = []
# erreurs_3 = []

# Nmax=1000
# u_anal = solve_C_v2_ter(Nmax,1,1,True)
# radii2 = np.linspace(0,R,Nmax)
# f_u_anal = scp.interpolate.interp1d(radii2, u_anal[:,0],'cubic')
# radii = np.linspace(0,R,Ntot)

# for Ntot in list_ntot:
# 	radii = np.linspace(0,R,Ntot)
# 	list_u_anal = []
# 	for r in radii:
# 		list_u_anal.append(f_u_anal(r))
# 	the_radii = np.linspace(0,R,Ntot)
# 	my_C = solve_C_v2_ter(Ntot,1,1,True)
# 	err1 = erreur_L1(my_C,list_u_anal,Ntot)
# 	err2 = erreur_L2(my_C,list_u_anal,Ntot)
# 	err3 = erreur_inf(my_C,list_u_anal,Ntot)
# 	erreurs_1.append(err1)
# 	erreurs_2.append(err2)
# 	erreurs_3.append(err3)


# list_h = []
# deltax_ref = R/Nmax
# for Ntot in list_ntot:
# 	deltax = R/Ntot
# 	list_h.append(deltax/deltax_ref)


# plt.figure()
# plt.loglog(list_h,erreurs_1, label = 'erreur L1',marker = '.')
# plt.loglog(list_h,erreurs_2, label = 'erreur L2',marker = '.')
# plt.loglog(list_h,erreurs_3, label = 'erreur infinie',marker = '.')
# plt.legend()
# plt.grid()
# plt.title('erreurs dans le cas stationnaire')
# plt.xlabel('epaisseur de maillage h [-]')
# plt.ylabel('erreurs [-]')
# plt.title("erreurs vs $h = \Delta x / \Delta x_{ref}$, equation stationnaire")
# #plt.show()



# ###################################
# #cas stationnaire, erreur en temps

# list_dts = [0.1,0.2,0.4,.8,1.0,2.0,10.]
# erreurs_1 = []
# erreurs_2 = []
# erreurs_3 = []

# Ntot = 500
# delta_t_min =1
# time_iter_min = Ttot/delta_t_min
# u_anal = solve_C_v2_ter(Ntot,delta_t_min,time_iter,True)
# radii2 = np.linspace(0,R,Ntot)
# radii = np.linspace(0,R,Ntot)

# for delta_t in list_dts:
# 	time_iter = Ttot/delta_t
# 	my_C = solve_C_v2_ter(Ntot,delta_t,time_iter,False)
# 	err1 = erreur_L1(my_C,u_anal,Ntot)
# 	err2 = erreur_L2(my_C,u_anal,Ntot)
# 	err3 = erreur_inf(my_C,u_anal,Ntot)
# 	erreurs_1.append(err1)
# 	erreurs_2.append(err2)
# 	erreurs_3.append(err3)



# plt.figure()
# plt.loglog(list_dts,erreurs_1, label = 'erreur L1',marker = '.')
# plt.loglog(list_dts,erreurs_2, label = 'erreur L2',marker = '.')
# plt.loglog(list_dts,erreurs_3, label = 'erreur infinie',marker = '.')
# plt.legend()
# plt.grid()
# plt.title('erreurs dans le cas instationnaire')
# plt.xlabel('epaisseur de maillage h [-]')
# plt.ylabel('erreurs [-]')
# plt.title("erreurs vs $\Delta t$, equation instationnaire")
# plt.show()
