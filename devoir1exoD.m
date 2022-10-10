% devoir 1: Vérification et validation en modélisation numérique
%devoir1, exercice D
Deff=10^(-10);  % le coefficient de diffusion effectif du sel
S=10^(-8);      %la quantité de sel 
K=4*10^(-9);    %constante de réaction
R=0.5;            %Rayon
D=1;            %diamètre
Ce=10;          %constante de la concentration en sel
%syms Ntot;         %Nombre des noeuds en symbolique
Ntot=5;

%paramètres du maillage
deltar=D/(Ntot-1);        %discretisation en espace
deltat=0.5;          %discretisation en temps
C_0=zeros(Ntot,1);

% définir les constantes qui entrent en jeu dans l'équation finale
beta=zeros(Ntot,1);
gamma=zeros(Ntot,1);
alpha=-deltat*Deff;
r=0;
for i=1:Ntot
r=r+deltar;
beta(i)=((deltar^2)+(deltat*Deff*deltar/r));
gamma(i)=alpha*(1-(deltar/r));
end
%betha=((deltar^2)+((deltat*deltar*Deff)/r))  %ajouter dans une boucles
%pour un deltar different 


%définir la matrice de résolution M
M=zeros(Ntot,Ntot);
M(1,1)=1;
M(Ntot,Ntot)=1;
for i=2:Ntot-1
    M(i,i-1)=alpha;
    M(i,i)=beta(i);
    M(i,i+1)=gamma(i);
end

%définir le terme additionel de droite
T=zeros(Ntot,1);
T(1,1)=Ce;
T(Ntot,1)=Ce;
for i=2:Ntot-1
    T(i,1)=deltar^2*(C_0(i)-S*deltat);
end


%calcul de la solution
t=0;
C=C_0;
for j=1:length(M)
    t=t+deltat;

    for i=2:Ntot-1
        T(i,1)=((deltar^2)*C(i))-(S*deltat*(deltar^2));
    end
    C=T.*inv(M)
end
disp(C)




 


