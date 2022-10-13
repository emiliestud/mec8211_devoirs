%devoir 1: Vérification et validation en modélisation numérique
%devoir1, exercice D

%Définitions des constantes

Deff = 10^(-10);                % le coefficient de diffusion effectif du sel
S = 10^(-8);                    %la quantité de sel 
K = 4*10^(-9);                  %constante de réaction
R = 1/2;                        %Rayon
D = 1;                          %diamètre
Ce = 10;                        %constante de la concentration en sel
A = 0;                          %facteur pour le cas stationnaire ou non A=0 stationnaire ,A=1 instationnaire
Ntot = 1000;
deltar = D/(Ntot-1);            %discretisation en espace
deltat = 1;                     %discretisation en temps
time = 1;

%%%%%%%%%%%%%%%%%%%%%%géneration de résultat%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C_1 = solveC1(Ntot,deltat,time);
% disp(C_1) 
% 
% C_2 = solveC2(Ntot,deltat,time);
% disp(C_2)

u_n = solveC2(Ntot,deltat,time);
u_a = definesolutionanalytique(Ntot);

%disp(u_a)
%disp(definesolutionanalytique(20))

 error1 = erreur1(u_n,u_a,Ntot);
 error2 = erreur2(u_n,u_a,Ntot);
 errorinfty = erreurinfty(u_n,u_a,Ntot);
 %disp(error1);
error2=[];
error1=[];
errorinfty=[];

for Ntot = 1:1000
  error1 = [error1 erreur1(u_n,u_a,Ntot)];
  error2 = [error2 erreur2(u_n,u_a,Ntot)];
  errorinfty = [errorinfty erreurinfty(u_n,u_a,Ntot)];
  %disp(error2);
 
end
loglog(linspace(1,Ntot,Ntot),error2);
xlabel('Nombre de noeud')
ylabel('Erreur')
legend('Erreur L2')
hold on
loglog(linspace(1,Ntot,Ntot),error1);
xlabel('Nombre de noeud')
ylabel('Erreur')
hold on
loglog(linspace(1,Ntot,Ntot),errorinfty);
xlabel('Nombre de noeud')
ylabel('Erreur')
legend('Erreur L2','Erreur L1','Erreur Linfinie')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Paramètres pour calcul EF

function C_0 = defineC_0(Ntot)
Ce = 10; 
C_0 = zeros(Ntot,1);
C_0(1,1) = Ce;

end

%Matrice de résolution pour le premier schéma de différenciation (Question D-E)

function M = definematrix1(Ntot,deltat,deltar)
     A = 0;                          %facteur pour le cas stationnaire ou non A=0 stationnaire ,A=1 instationnaire
     Deff = 10^(-10); 
     alpha = -deltat*Deff;
     r = 0;
     M = zeros(Ntot,Ntot);
     M(1,2) = 1.0;
     M(Ntot,Ntot) = 1.0;
     
     for i=2:Ntot-1        %i=2,3,4
             
          r = r + deltar;
          M(i,i-1) = alpha;
          M(i,i) = ((A*(deltar^2))-alpha*(2+(deltar/r)));
          M(i,i+1) = alpha*(1+(deltar/r));

     end
end




%définir le terme additionel de droite

function D = defineadditionalvector(Ntot,C,deltat,deltar)
    A = 0;                          %facteur pour le cas stationnaire ou non A=0 stationnaire ,A=1 instationnaire
    S = 10^(-8); 
    r = 0;
    Ce = 10; 
    D = zeros(Ntot,1);
    %D(1,1) = Ce;
    D(Ntot,1) = Ce;
    for i=2:Ntot-1
            r = r+deltar;
            D(i,1) = A*deltar^2*C(i)-S*deltat*deltar^2;
    end
end


%calcul de la solution à chaque itération de temps

    function C = solveC1(Ntot,deltat,time)
        R = 0.5;
        A = 0;
        t = 0;
        j = 0;
        deltar = R/(Ntot-1);
        C = defineC_0(Ntot);
        M_1 = definematrix1(Ntot,deltat,deltar);
        
        while j < time
            t = t+deltat;
            D_1 = defineadditionalvector(Ntot,C,deltat,deltar);
            C = linsolve(M_1,D_1);
            j=j+1;
         
        end
          
    end

%Matrice de résolution pour le deuxième schéma de différenciation (Question F)
   
function M = definematrix2(Ntot,deltat,deltar)
     A = 0;                          %facteur pour le cas stationnaire ou non A=0 stationnaire ,A=1 instationnaire
     Deff = 10^(-10); 
     alpha = -deltat*Deff;
     r = 0;
     M = zeros(Ntot,Ntot);
     M(1,2) = 4/3;
     M(1,3) = -1/3;
     M(Ntot,Ntot) = 1.0;
     
     for i=2:Ntot-1        %i=2,3,4
             
          r = r + deltar;
          M(i,i-1) = alpha*(1-deltar/(2*r));
          M(i,i) = A*(deltar^2)-2*alpha;
          M(i,i+1) = alpha*(1+(deltar/(2*r)));

     end
end

%calcul de la solution à chaque itération de temps
  function C = solveC2(Ntot,deltat,time)
        R = 0.5;
        A = 0;
        t = 0;
        j = 0;
        deltar = R/(Ntot-1);
        C = defineC_0(Ntot);
        M_2 = definematrix2(Ntot,deltat,deltar);
        
        while j < time
            t = t+deltat;
            D_2 = defineadditionalvector(Ntot,C,deltat,deltar);
            C = linsolve(M_2,D_2);
            j=j+1;
         
        end
          
  end





  %%%%%%%%%%%%%%%%%%Solution analytique stationnaire-question C %%%%%%%%%%%%%%%

  function u_a = definesolutionanalytique(Ntot)
 
  Deff = 10^(-10);                % le coefficient de diffusion effectif du sel
  S = 10^(-8);                    %la quantité de sel 
  R = 1/2;                        %Rayon
  Ce = 1;
  u_a = zeros(Ntot,1);  
  r = linspace(0,R,Ntot);
  for i = 1:Ntot
      r1=r(i);
      u_a(i)=0.25*(S/Deff)*(R^2)*(((r1^2/R^2)-1))+Ce;
  end
  
  end
  


  %%%%%%%%%%%%%%%%%%%% Calcul de l'erreur L1 %%%%%%%%%%%%%%%%%%%%%%%%%
  function err1 = erreur1(u_n,u_a,Ntot)
  err1 = 0;
  R = 0.5;
  r = 0;
  deltar = R/(Ntot-1);
  for k=1:Ntot
      r = r+deltar;
      err1 = sum(abs((u_n(k))-(u_a(k))));
  end
  err1 = err1/Ntot;
  end

  %%%%%%%%%%%%%%%%%%%% Calcul de l'erreur L2 %%%%%%%%%%%%%%%%%%%%%%%%%
  function err2 = erreur2(u_n,u_a,Ntot)
  err2 = 0;
  r = 0;
  R = 0.5;
  deltar = R/(Ntot-1);
  for k=1:Ntot
      r = r+deltar;
      %err2 = err2+r+abs(u_n(k)-u_a(k))^2;
      err2=sqrt(sum(sum(((u_n(k)-u_a(k))^2))));
  end
  err2 = err2/Ntot;
  err2 = sqrt(err2);
  end


  %%%%%%%%%%%%%%%%%%%% Calcul de l'erreur Linfty %%%%%%%%%%%%%%%%%%%%%%%%%
  
  function errinfty = erreurinfty(u_n,u_a,Ntot)
  errinfty = 0;
 
  for k=1:Ntot
%       if abs(u_n(k)-u_a(k)) >=errinfty
%       errinfty = norm((u_n(k)-u_a(k)),Inf);
%       end

   errinfty = abs(u_n(k)-u_a(k));
  end
  end

  
