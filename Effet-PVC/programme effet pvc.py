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
    F = LectureApp("MAI-1m.txt")   
    G = LectureApp("MAI-70cm.txt")
    H = LectureApp("MAI-35cm.txt")
    I = LectureApp("MAI-10cm.txt")
    J = LectureApp("MAI-Surface.txt")
    
    for X in [F,G,H,I,J] :
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
    print("Coefficient 1 - h=100 : " +str(coeff(F)))
    print("Coefficient 2 - h=70 : " +str(coeff(G)))
    print("Coefficient 3 - h=35 : " +str(coeff(H)))
    print("Coefficient 4 - h=10 : " +str(coeff(I)))
    print("Coefficient 5 - h=10 : " +str(coeff(J)))
    moycoef2 = (coeff(F)+coeff(G)+coeff(H)+coeff(I)+coeff(J))/5
    print("Moyenne des coefficients dans le PVC : " + str(moycoef2))
    
    for X in [F,G,H,I,J] :
        for i in range(len(X[0])) : 
            X[0][i]=X[0][i]/3600
    
    #Affichage des courbes brutes en fonction du temps
    plt.clf()
    plt.scatter(F[0],F[1], marker='+', s=1, c='red', label="h=100")
    plt.scatter(G[0],G[1], marker='+', s=1, c='blue', label="h=70")
    plt.scatter(H[0],H[1], marker='+', s=1, c='green', label="h=35")
    plt.scatter(I[0],I[1], marker='+', s=1, c='maroon', label="h=10")
    plt.scatter(J[0],J[1], marker='+', s=1, c='black', label="h=0")
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Dérive (en mètres)")
    plt.title("Dérives en statique, PVC")
    plt.legend(loc=0)
    plt.savefig("Dérives en statique, PVC.png", dpi=500)
    plt.clf()
    
    fh=coeff(F) #m/h
    gh=coeff(G) #m/h
    hh=coeff(H) #m/h
    ih=coeff(I) #m/h
    jh=coeff(J) #m/h
    
    # Régressions linéaires :
    plt.plot(plotage(F)[0], plotage(F)[1], color='red', label="h=100,a="+str(ceil(10*fh)/10))
    plt.plot(plotage(G)[0], plotage(G)[1], color='blue', label="h=70,  a="+str(floor(10*gh)/10))
    plt.plot(plotage(H)[0], plotage(H)[1], color='green', label="h=35,  a="+str(floor(10*hh)/10))
    plt.plot(plotage(I)[0], plotage(I)[1], color='maroon', label="h=10,  a="+str(floor(10*ih)/10))
    plt.plot(plotage(J)[0], plotage(J)[1], color='black', label="h=0,    a="+str(ceil(10*jh)/10))
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Distance (en mètres)")
    plt.title("régressions linéaires dans le PVC")
    plt.legend(loc=2)
    plt.savefig("regs lineaires PVC.png", dpi=500)

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