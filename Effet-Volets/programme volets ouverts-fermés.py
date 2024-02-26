# Importation des bibliothèques utiles à notre traitement de données 
from math import * 

# Numpy pour les matrices
import numpy as np

# MatplotLib pour les graphiques
import matplotlib.pyplot as plt

# Scipy Stats pour réaliser des estimations par densité de noyau
from scipy.stats import kde

# Un module annexe d'estimation par noyau pour régler une contrainte
from statsmodels.nonparametric.kde import KDEUnivariate

# Un module utile pour les représentations doubles des estimations par densité de noyau
import seaborn as sns

# Lecture des données issues des relevés GPS sous forme .txt
# Avec la création de différents tableaux de données
# Calcul des intervalles de distances, des intervalles de temps, et autres


def LectureApp(string):
    
    # Lecture du document
    texte  = open(string, "r")
    t = texte.readlines()
    a = ""
    
    # On récupère le nombre de lignes
    for i in range(len(t)):
        a += t[i]
    
    # Ensuite, avec toutes les fonctions suivantes, on va extraire les 
    # données qui nous intéressent.
    
    #fonction qui cherche si le 'motif' est à la position 'i' dans le 'texte'
    def est_ici(texte,motif,i):
        a = 0
        if len(motif)>len(texte):
            return False
    
        while a<len(motif):
            if texte[i+a] == motif[a]:
                a += 1
            else:
                return False
        return True
    
    #fonction qui cherche si le 'motif' est dans le 'texte'
    def est_sous_mot(texte,motif):
        indice = 0
        while indice < len(texte):
            if est_ici(texte,motif,indice) == False :
                indice += 1
            else:
                return True
        return False
    
    #fonction qui cherche la(les) position(s) du motif 'i' dans le 'texte'
    def position_sous_mot(texte,motif):
        occurence = []
        for i in range (len(texte)):
            if est_ici(texte, motif, i) ==True :
                occurence.append(i)
        return(occurence)
    
    #fonction qui donne la liste contenant les différentes positions des motifs 
    # 'c' et 'b' dans le texte
    def liste_pour_recherche_de_mot(c,b):
        D = position_sous_mot(a,c) #position(s) du motif 'c' dans le texte 'a'
        E = position_sous_mot(a,b) #position(s) du motif 'b' dans le texte 'a'
        n = len(c)
        for i in range(len(D)):
            D[i] += n
        return D,E
    
    liste = liste_pour_recherche_de_mot("<DistanceMeters>","</DistanceMeters>")
    liste2 = liste_pour_recherche_de_mot("<Time>","</Time")
    
    def entre_occurence (l1,l2):
        L = []
        for i in range(len(l1)):
            longueur = l2[i]-l1[i]
            mot = ""
            for j in range(longueur):
                mot += a[l1[i]+j]
            L.append(mot)
        return L
    
    def intervalle_temps(liste):
        L = []
        for i in range(len(liste)-1):
            donnees1 = liste[i].split('T')
            donnees2 = liste[i+1].split('T')
            donnees3 = donnees1[1].split(".")
            donnees4 = donnees2[1].split(".")
            donnees5 = donnees3[0].split(":")
            donnees6 = donnees4[0].split(":")
            if donnees1[0] == donnees2[0]:
                L.append((int(donnees6[0])*3600+int(donnees6[1])*60+int(donnees6[2]))-(int(donnees5[0])*3600+int(donnees5[1])*60+int(donnees5[2])))
            else:
                L.append((int(donnees6[0])*3600+int(donnees6[1])*60+int(donnees6[2])+3600*24)-(int(donnees5[0])*3600+int(donnees5[1])*60+int(donnees5[2])))
        return L
    
    def intervalle_distance(liste):
        L = []
        for i in range(1,len(liste)-1):
            L.append(float(liste[i+1])-float(liste[i]))
        return L
    
    intervale_distance = intervalle_distance(entre_occurence(liste[0],liste[1]))
    intervale_temps = intervalle_temps(entre_occurence(liste2[0],liste2[1]))
    
    p = []
    m = []
    somme = 0
    somme_bis = 0
    temoin = []
    
    #condition : ne les prendra pas en compte dans les expériences statiques :
    
    #expérience de 60min : 
    if len(intervale_distance)==1037 or len(intervale_distance)==1269 or len(intervale_distance)==1576 :
        vmoy = 5.2390465693 
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
     
    #expérience de 40min :     
    elif len(intervale_distance)==674 or len(intervale_distance)==775 or len(intervale_distance)==1121 :
        vmoy = 5.1399026764
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
    #expérience de 20 min :      
    elif len(intervale_distance)==336 or len(intervale_distance)==426 or len(intervale_distance)==518 :
        vmoy = 5.25497512438
        for i in range(len(intervale_temps)):
            somme += intervale_distance[i]
            somme_bis += intervale_temps[i]
            p.append(somme)
            m.append(somme_bis)
            if abs(somme-(vmoy*somme_bis)) < 1000 :
                temoin.append(abs(somme-(vmoy*somme_bis)))
            if abs(somme-(vmoy*somme_bis)) > 1000 :
                m.pop()
                p.pop()
   
    #expérience statique : 
    else:
        for i in range(len(intervale_distance)):
            somme += intervale_distance[i]
            p.append(somme)
        for i in range(len(intervale_temps)):    
            somme_bis += intervale_temps[i]
            m.append(somme_bis)

        
    # On met dans une nouvelle liste les données suivantes sous forme de listes :
    K=[]
    # temps total
    K.append(m)
    # distance totale
    K.append(p)
    # intervalle temps
    K.append(intervale_temps)
    # intervalle distance
    K.append(intervale_distance)
    # moyenne de la distance parcourue par seconde
    K.append(temoin)
    
    return(K)



# Commande pour obtenir les données relatives à l'expérience dynamique 
# de 20 minutes pour le capteur Garmin 735 XT (montre) :
    
# print(LectureApp("ACT-BELDYN-20m-GARMIN735.txt"))



def mesure_des_ecarts_statique():
    
    #Indication dans la console de quelle partie il s'agit
    print("MESURE DES ÉCARTS STATIQUES \n")
    
    #Récupération des listes de données des différents fichiers.
    A = LectureApp("059.txt")
    B = LectureApp("101.txt")
    C = LectureApp("104.txt")  
    D = LectureApp("105.txt") 
    E = LectureApp("106.txt") 
    F = LectureApp("107.txt")   
    G = LectureApp("108.txt")
    H = LectureApp("131.txt")
    I = LectureApp("142.txt")
    J = LectureApp("209.txt")
    K = LectureApp("320.txt")
    L = LectureApp("625 vf.txt")
    M = LectureApp("711 vf.txt")
    N = LectureApp("753 vf.txt") 
    O = LectureApp("821 vf.txt") 
    P = LectureApp("826 vf.txt")
    Q = LectureApp("913 vf.txt")
    R = LectureApp("1205 vf.txt")
    
    for X in [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R] :
        if len(X[2]) > len(X[3]) : 
            while len(X[2]) != len(X[3]) :
                X[2].pop()
        if len(X[3]) > len(X[2]) : 
            while len(X[2]) != len(X[3]) :
                X[3].pop()
        if len(X[0]) > len(X[1]) : 
            while len(X[0]) != len(X[1]) :
                X[0].pop()
        if len(X[1]) > len(X[0]) : 
            while len(X[0]) != len(X[1]) :
                X[1].pop()
        
    
    # Informations utiles
    print(" -------- Informations Utiles -------- \n")
    print("Nombre de mesures (1h25 volets ouverts) : "+str(len(D[0])))

    cana1 = []+A[4]+B[4]+C[4]+D[4]+E[4]+F[4]+G[4]+H[4]+I[4]+J[4]+K[4]
    cana1.sort()
    pana1=[]+A[1]+B[1]+C[1]+D[1]+E[1]+F[1]+G[1]+H[1]+I[1]+J[1]+K[1]
    pana1tps = []+A[0]+B[0]+C[0]+D[0]+E[0]+F[0]+G[0]+H[0]+I[0]+J[0]+K[0]
    pana1.sort()
    pana1tps.sort()
    (coe1,ordo1) = np.polyfit(pana1tps, pana1, 1)
    print("Coefficient général : "+str(coe1))
    print("Coefficient 1 - volets ouverts : " +str(coeff(A)))
    print("Coefficient 2 - volets ouverts : " +str(coeff(B)))
    print("Coefficient 3 - volets ouverts : " +str(coeff(C)))
    print("Coefficient 4 - volets ouverts : " +str(coeff(D)))
    print("Coefficient 5 - volets ouverts : " +str(coeff(E)))
    print("Coefficient 6 - volets ouverts : " +str(coeff(F)))
    print("Coefficient 7 - volets ouverts : " +str(coeff(G)))
    print("Coefficient 8 - volets ouverts : " +str(coeff(H)))
    print("Coefficient 9 - volets ouverts : " +str(coeff(I)))
    print("Coefficient 10 - volets ouverts : " +str(coeff(J)))
    print("Coefficient 11 - volets ouverts : " +str(coeff(K)))
    moycoef1 = (coeff(A)+coeff(B)+coeff(C)+coeff(D)+coeff(E)+coeff(F)+coeff(G)+coeff(H)+coeff(I)+coeff(J)+coeff(K))/11
    print("Moyenne des coefficients volets ouverts : " + str(moycoef1))
    
    cana2 = []+L[4]+M[4]+N[4]+O[4]+P[4]+Q[4]+R[4]
    cana2.sort()
    pana2=[]+L[1]+M[1]+N[1]+O[1]+P[1]+Q[1]+R[1]
    pana2tps = []+L[0]+M[0]+N[0]+O[0]+P[0]+Q[0]+R[0]
    pana2.sort()
    pana2tps.sort()
    (coe2,ordo2) = np.polyfit(pana2tps, pana2, 1)
    print("Coefficient général : "+str(coe2))
    print("Coefficient 1 - volets fermés : " +str(coeff(L)))
    print("Coefficient 2 - volets fermés : " +str(coeff(M)))
    print("Coefficient 3 - volets fermés : " +str(coeff(N)))
    print("Coefficient 4 - volets fermés : " +str(coeff(O)))
    print("Coefficient 5 - volets fermés : " +str(coeff(P)))
    print("Coefficient 6 - volets fermés : " +str(coeff(Q)))
    print("Coefficient 7 - volets fermés : " +str(coeff(R)))
    moycoef2 = (coeff(L)+coeff(M)+coeff(N)+coeff(O)+coeff(P)+coeff(Q)+coeff(R))/7
    print("Moyenne des coefficients volets fermés : " + str(moycoef2))
    
    for X in [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R] :
        for i in range(len(X[0])) : 
            X[0][i]=X[0][i]/3600
    
    #Affichage des courbes brutes en fonction du temps
  
    plt.clf()

    plt.scatter(H[0],H[1], marker='+', s=1, c='black')
    plt.scatter(I[0],I[1], marker='+', s=1, c='black')
    plt.scatter(J[0],J[1], marker='+', s=1, c='black')
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Dérive (en mètres)")
    plt.title("Dérives en statique en intérieur, volets ouverts")
    plt.savefig("Dérives en statique en intérieur, volets ouverts.png", dpi=500)
    
    plt.clf()
    plt.scatter(L[0],L[1], marker='+', s=1, c='black')
    plt.scatter(M[0],M[1], marker='+', s=1, c='black')
    plt.scatter(N[0],N[1], marker='+', s=1, c='black')
    plt.scatter(O[0],O[1], marker='+', s=1, c='black')
    plt.scatter(P[0],P[1], marker='+', s=1, c='black')
    plt.scatter(Q[0],Q[1], marker='+', s=1, c='black')
    plt.scatter(R[0],R[1], marker='+', s=1, c='black')
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Dérive (en mètres)" )
    plt.title("Dérives en statique en intérieur, volets fermés")
    plt.savefig("Dérives en statique en intérieur, volets fermés.png", dpi=500)

    # Affichage de l'estimation de densité par noyau (KDE) des intervalles de temps
   
    # Calcul des valeurs moyennes
    #volets ouverts
    m1=moycoef1*3600 #en m/h
    t1=[]
    l1=[]
    for i in range(1000):
        t1.append(i*11/1000)    
        l1.append(i*11*m1/1000000) #repasser en km
    mh1=floor(m1)
    plt.plot(t1, l1, color='blue', label="volets ouverts, coef="+str(mh1)+"m/h")
    
    #volets fermés 
    m2=moycoef2*3600 #en m/h
    t2=[]
    l2=[]
    for i in range(1000) :
        t2.append(i*11/1000)    
        l2.append(i*11*m2/1000000) #repasser en km
    mh2=ceil(m2)
    plt.plot(t2, l2, color='red', label="volets fermés,  coef="+str(mh2)+"m/h")
    
    # Mise en forme
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Dérive (en km)")
    plt.title("régressions linéaires et leur moyenne")
    plt.legend(loc=2)
    plt.savefig("régression linéaires pour l'expérience des volets.png", dpi=500)

    return(None)
    
    
    
    
# Définition de plusieurs fonctions Annexes utiles pour nos fonctions principales


# Calcul de coefficient par régression linéaire
def coeff(L):
    tps = L[0]
    parc= L[1]
    coef, ordo= np.polyfit(tps, parc, 1)
    return(coef)


# Construction de deux listes pour exploiter la régression linéaire dans des graphiques
def plotage(L):
    coef = coeff(L)
    t=[]
    mg=[]
    for i in (L[0]):
        t.append(i)
        mg.append(i*coef)
    K=[]
    K.append(t)
    K.append(mg)
    return(K)


# Première fonction d'estimation de densité par noyau gaussien utilisée
def GAUSSIENNE(b):
    density = kde.gaussian_kde(b)
    x = np.linspace(-5,45, 400)
    y=density(x)
    return(x,y)

# Deuxième fonction d'estimation de densité par noyau gaussien utilisée
def GAUSSIENNE2(b):
    density = kde.gaussian_kde(b)
    x = np.linspace(-2,8, 400)
    y=density(x)
    return(x,y)

# Fonction annexe du calcul de KDE (car on a rencontré quelques problèmes avec le module de base)
def kde_statsmodels_u(x, x_grid, bandwidth=3, **kwargs):
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)




# FONCTION DE LANCEMENT DU PROGRAMME 

def Start():
    mesure_des_ecarts_statique()

Start()