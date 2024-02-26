# Importation des bibliothèques utiles à notre traitement de données 

# Pour les maths 
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
            somme_bis += intervale_temps[i]
            p.append(somme)
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
    A = LectureApp("ACT22-FINAL.txt")
    B = LectureApp("ACT19-FINAL.txt")
    C = LectureApp("ACT10-FINAL.txt")
    D = LectureApp("ACT3-FINAL.txt")
    E = LectureApp("ACT1-FINAL.txt")
    
    
    # Informations utiles
    print(" -------- Informations Utiles -------- \n")
    print("Nombre de mesures (22h) : "+str(len(A[0])))
    print("Nombre de mesures (1h) : "+str(len(E[0])))
    cana = []+A[4]+B[4]+C[4]+D[4]+E[4]
    cana.sort()
    pana=[]+A[1]+B[1]+C[1]+D[1]+E[1]
    panatps = []+A[0]+B[0]+C[0]+D[0]+E[0]
    pana.sort()
    panatps.sort()
    (coe,ordo) = np.polyfit(panatps, pana, 1)
    print("Coefficient général : "+str(coe))
    print("Coefficient 1h : " +str(coeff(E)))
    print("Coefficient 3h : " +str(coeff(D)))
    print("Coefficient 10h : " +str(coeff(C)))
    print("Coefficient 19h : " +str(coeff(B)))
    print("Coefficient 22h : " +str(coeff(A)))
    moycoef = (coeff(A)+coeff(B)+coeff(C)+coeff(D)+coeff(E))/5
    print("Moyenne des coefficients : " + str(moycoef))
    
    
    #Affichage des courbes brutes en fonction du temps
    
    Z=LectureApp("ACT1-FINAL.txt")
    for i in range (len(Z[0])) :
        Z[0][i] = Z[0][i]/3600
    for i in range (len(Z[1])) :
        Z[1][i] = Z[1][i]/1000
    
    Y=LectureApp("ACT3-FINAL.txt")
    for i in range (len(Y[0])) :
        Y[0][i] = Y[0][i]/3600
    for i in range (len(Y[1])) :
        Y[1][i] = Y[1][i]/1000
    
    X=LectureApp("ACT10-FINAL.txt")
    for i in range (len(X[0])) :
        X[0][i] = X[0][i]/3600
    for i in range (len(X[1])) :
        X[1][i] = X[1][i]/1000
    
    W=LectureApp("ACT19-FINAL.txt")
    for i in range (len(W[0])) :
        W[0][i] = W[0][i]/3600
    for i in range (len(W[1])) :
        W[1][i] = W[1][i]/1000
    
    V=LectureApp("ACT22-FINAL.txt")
    for i in range (len(V[0])) :
        V[0][i] = V[0][i]/3600
    for i in range (len(V[1])) :
        V[1][i] = V[1][i]/1000
    
    
    plt.clf()
    plt.scatter(V[0],V[1], marker='+', s=1, c='blue', label="22h")
    plt.legend(loc=0)
    plt.xlabel("durée (en heures)")
    plt.ylabel("dérive (en km)")
    plt.title("Dérives du capteur GPS en statique ")
    plt.savefig("Représentation des dérives 1 courbe.png", dpi=500)
    
    plt.clf()
    plt.scatter(V[0],V[1], marker='+', s=1, c='blue', label="22h")
    plt.scatter(W[0],W[1], marker='+', s=1, c='red', label="19h")
    plt.scatter(X[0],X[1], marker='+', s=1, c='green', label="10h")
    plt.scatter(Y[0],Y[1], marker='+', s=1, c='yellow', label="3h")
    plt.scatter(Z[0],Z[1], marker='+', s=1, c='purple', label="1h")
    plt.legend(loc=0)
    plt.xlabel("durée (en heures)")
    plt.ylabel("dérive (en km)")
    plt.title("Dérives du capteur GPS en statique ")
    plt.savefig("Représentation des dérives.png", dpi=500)
    
    
    #Affichage de la répartition des intervalles distance en fonction des intervalles de temps 
    plt.clf()
    plt.scatter(A[2],A[3], marker='.', s=1, c='blue', label="22 heures")
    plt.scatter(B[2],B[3], marker='.', s=1, c='red', label="19 heures")
    plt.scatter(C[2],C[3], marker='.', s=1, c='green', label="10 heures")
    plt.scatter(D[2],D[3], marker='.', s=1, c='yellow', label="3 heures")
    plt.scatter(E[2],E[3], marker='.', s=1, c='purple', label="1 heure")
    plt.title("Répartition des intervalles de distances selon l'intervalle de temps")
    plt.xlabel("Intervalles de temps (en secondes)")
    plt.ylabel("Dérive pendant l'intervalle de temps (en mètres)", fontsize=8)
    plt.legend(loc=1)
    plt.savefig("ACT-Représentation Répartition distance-temps.png", dpi=500)
    
    # Affichage de l'estimation de densité par noyau (KDE) des intervalles de distance
    
    # Estimation de densité par noyau (via fonction auxiliaire)
    AL = GAUSSIENNE(A[3])
    BL = GAUSSIENNE(B[3])
    CL = GAUSSIENNE(C[3])
    DL = GAUSSIENNE(D[3])
    EL = GAUSSIENNE(E[3])
    
    # Affichage
    plt.clf()
    plt.plot(AL[0], AL[1], color="blue", label="22 heures")
    plt.plot(BL[0], BL[1], color="red", label="19 heures")
    plt.plot(CL[0], CL[1], color="green", label="10 heures")
    plt.plot(DL[0], DL[1], color="yellow", label="3 heures")
    plt.plot(EL[0], EL[1], color="purple", label="1 heure")
    plt.title("densité statistique des intervalles de distances mesurés")
    plt.legend(loc=1)
    plt.xlim(-5, 25)
    plt.xlabel("Intervalle de distance (en mètres)")
    plt.ylabel("Probabilité effective (0.4 = 40%)")
    plt.savefig("ACT-GAUSSIENNE-DISTANCE.png", dpi=500)
    
    
    # Affichage de l'estimation de densité par noyau (KDE) des intervalles de temps
    
    # Estimation de densité par noyau (via fonction auxiliaire)
    AK = GAUSSIENNE(A[2])
    BK = GAUSSIENNE(B[2])
    CK = GAUSSIENNE(C[2])
    DK = GAUSSIENNE(D[2])
    EK = GAUSSIENNE(E[2])
    
    # Affichage
    plt.clf()
    plt.plot(AK[0], AK[1], color="blue", label="22 heures")
    plt.plot(BK[0], BK[1], color="red", label="19 heures")
    plt.plot(CK[0], CK[1], color="green", label="10 heures")
    plt.plot(DK[0], DK[1], color="yellow", label="3 heures")
    plt.plot(EK[0], EK[1], color="purple", label="1 heure")
    plt.title("densité statistique des intervalles de temps mesurés")
    plt.legend(loc=1) 
    plt.xlabel("Intervalle de temps (en secondes)")
    plt.ylabel("Probabilité effective (0.1 = 10%)")
    plt.savefig("ACT-GAUSSIENNE-TEMPS.png", dpi=500)
    
    
    # Graphique représentant les régressions linéaires et leur courbe associée    
    plt.clf()
    # regression linéaire
    number=coeff(V)
    ah=(int(str(number*1000)[:3])) #=459 m/h
    plt.plot(plotage(V)[0], plotage(V)[1], color='blue', label="y =ax, a="+str(ah)+"m/h")
    #courbes réelles
    plt.scatter(V[0],V[1], marker='+', s=1, c='blue', alpha=0.05, label="22 heures")
    plt.xlabel("temps (en heures)")
    plt.ylabel("dérive (en km)")
    plt.legend(loc=0) 
    plt.title("régression linéaire et courbe réelle")
    plt.savefig("reg linéaire avec courbe réelle.png", dpi=500)
    
    
    plt.clf()
    # Régressions linéaires :
    bh=(int(str(coeff(W)*10000)[:3])) #= 315 m/h
    ch=(int(str(coeff(X)*1000)[:3])) #=363 m/h
    dh=(int(str(coeff(Y)*10000)[:3])) #=352 m/h
    eh=(int(str(coeff(Z)*1000)[:3])) #=453 m/h
    
    plt.plot(plotage(Z)[0], plotage(Z)[1], color='purple', label="1 heure,     a="+str(ceil(eh)))
    plt.plot(plotage(Y)[0], plotage(Y)[1], color='yellow', label="3 heures,   a="+str(ceil(dh)))
    plt.plot(plotage(X)[0], plotage(X)[1], color='green', label="10 heures, a="+str(ceil(ch)))
    plt.plot(plotage(W)[0], plotage(W)[1], color='red', label="19 heures, a="+str(ceil(bh)))
    plt.plot(plotage(V)[0], plotage(V)[1], color='blue', label="22 heures, a="+str(ceil(ah)))
    
    # Courbes réelles :
    plt.scatter(V[0],V[1], marker='+', s=1, c='blue', alpha=0.05)
    plt.scatter(W[0],W[1], marker='+', s=1, c='red', alpha=0.05)
    plt.scatter(X[0],X[1], marker='+', s=1, c='green', alpha=0.05)
    plt.scatter(Y[0],Y[1], marker='+', s=1, c='yellow', alpha=0.05)
    plt.scatter(Z[0],Z[1], marker='+', s=1, c='purple', alpha=0.05)
    
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Distance (en km)")
    plt.title("régressions linéaires et courbes réelles")
    plt.legend(loc=2)
    plt.savefig("regs linéaires avec courbes réelles.png", dpi=500)
    
    
    #Affichage des régressions linéaires avec la moyenne
    plt.clf()
    plt.plot(plotage(Z)[0], plotage(Z)[1], color='purple', label="1 heure,     a="+str(ceil(eh)))
    plt.plot(plotage(Y)[0], plotage(Y)[1], color='yellow', label="3 heures,   a="+str(ceil(dh)))
    plt.plot(plotage(X)[0], plotage(X)[1], color='green', label="10 heures, a="+str(ceil(ch)))
    plt.plot(plotage(W)[0], plotage(W)[1], color='red', label="19 heures, a="+str(ceil(bh)))
    plt.plot(plotage(V)[0], plotage(V)[1], color='blue', label="22 heures, a="+str(ceil(ah)))
    
    
    # Calcul des valeurs moyennes
    mh=moycoef*3600/1000 #en km/h
    time=[]
    long=[]
    for i in range(1000):
        time.append(i*22/1000)    
        long.append(i*22*mh/1000)
    mhh=str(moycoef*3600)[:3]
    plt.plot(time, long, color='black', label="moyenne,  a="+str(mhh))
    
    
    # Mise en forme
    plt.xlabel("Temps (en heures)")
    plt.ylabel("Dérive (en km)")
    plt.title("Régressions linéaires et la moyenne")
    plt.legend(loc=2)
    plt.savefig("ACT-Courbes régressions linéaires + moyenne.png", dpi=500)

    return(None)

# Deuxième expérience


def mesure_des_ecarts_dynamique():
    
    #Indication dans la console de quelle partie il s'agit
    print("\n \n MESURE DES ÉCARTS DYNAMIQUES \n")
    
    #Récupération des listes de données des différents fichiers.
    A = LectureApp("ACT-BELDYN-1h-COMPTEURGARMIN.txt")
    B = LectureApp("ACT-BELDYN-1h-GARMIN735.txt")
    C = LectureApp("ACT-BELDYN-1h-GARMIN935.txt")
    D = LectureApp("ACT-BELDYN-40m-COMPTEURGARMIN.txt")
    E = LectureApp("ACT-BELDYN-40m-GARMIN735.txt")
    F = LectureApp("ACT-BELDYN-40m-GARMIN935.txt")
    G = LectureApp("ACT-BELDYN-20m-COMPTEURGARMIN.txt")
    H = LectureApp("ACT-BELDYN-20m-GARMIN735.txt")
    I = LectureApp("ACT-BELDYN-20m-GARMIN935.txt")


    # Informations utiles
    print(" -------- Informations Utiles (Expérience Dynamique) -------- \n")
    print("Nombre de mesures (Compteur Garmin 1h) : "+str(len(A[0])))
    print("Nombre de mesures (Garmin 735 1h) : "+str(len(B[0])))
    print("Nombre de mesures (Garmin 935 1h) : "+str(len(C[0]))+"\n")
    
    
    # Calcul du coefficient directeur général (i.e. vitesse globale)
    
    distance_totale=[]+A[1]+B[1]+C[1]+D[1]+E[1]+F[1]+G[1]+H[1]+I[1]
    temps_total = []+A[0]+B[0]+C[0]+D[0]+E[0]+F[0]+G[0]+H[0]+I[0]
    distance_totale.sort()
    temps_total.sort()
    (vitesse,ordo) = np.polyfit(temps_total, distance_totale, 1)
    print("Coefficient général (vitesse en mètres/seconde): "+str(vitesse))
    
    
    
    # Représentation de la répartition des intervalles de distance et de temps
    
    # Création de l'étalement des abscisses
    x_grid = np.linspace(-2, 30, 1000)    
    
    # Reset du cadre graphique
    plt.clf()
    
    # Représentation des intervalles de distance
    
    # PLOT de la répartition de densité distance sous forme gaussiennes
    AL = GAUSSIENNE(A[3])
    #BL = GAUSSIENNE(B[3])
    CL = GAUSSIENNE(C[3])
    DL = GAUSSIENNE(D[3])
    #EL = GAUSSIENNE(E[3])
    FL = GAUSSIENNE(F[3])
    GL = GAUSSIENNE(G[3])
    #HL = GAUSSIENNE(H[3])
    #IL = GAUSSIENNE(I[3])
    
    
    # Pour certaines listes, on a utilisé un module annexe car nous avions des problèmes 
    # que nous n'avons pas réussi à identifier
    BL = [x_grid, kde_statsmodels_u(B[3], x_grid, bandwidth=1)]
    EL = [x_grid, kde_statsmodels_u(E[3], x_grid, bandwidth=1)]
    HL = [x_grid, kde_statsmodels_u(H[3], x_grid, bandwidth=1)]
    IL = [x_grid, kde_statsmodels_u(I[3], x_grid, bandwidth=1)]
        
    
    # On ajoute chaque répartition gaussienne
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(CL[0], CL[1], color="green", label="935 XT")
    ax.plot(FL[0], FL[1], "g--")
    ax.plot(IL[0], IL[1], 'g:')
    maxline, = ax.plot(0, 0, '-', color='green')
    minline, = ax.plot(0, 0, '--', color='green')
    basicline, = ax.plot(0, 0, ':', color='green')
    # On ajoute la première légende
    leg1 = ax.legend(loc='upper right', fontsize=10)
    # On met ensuite la deuxième légende
    leg2 = ax.legend([maxline,minline, basicline],['1h','40m', '20m'], loc='right', fontsize=10)
    # Et on fait revenir la première légende (astuce pour légender deux fois)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.ylim(0, 0.3)
    plt.xlim(-1, 28)
    plt.title("Densité statistique des intervalles de distance")
    plt.xlabel("Intervalle de distance (en mètres)")
    plt.ylabel("Probabilité effective (0.4 = 40%)")
    plt.savefig("ACT-GAUSIENNE-DYNAMIQUE-DISTANCE-935-XT.png", dpi=500)
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(AL[0], AL[1], color="blue", label="EDGE 1000")
    ax.plot(BL[0], BL[1], color="red", label="735 XT")
    ax.plot(CL[0], CL[1], color="green", label="935 XT")
    ax.plot(DL[0], DL[1], "b--")
    ax.plot(EL[0], EL[1], 'r--')
    ax.plot(FL[0], FL[1], 'g--')
    ax.plot(GL[0], GL[1], 'b:')
    ax.plot(HL[0], HL[1], 'r:')
    ax.plot(IL[0], IL[1], 'g:')    
    # Réglages des paramètres
    maxline, = ax.plot(0, 0, '-', color='blue')
    minline, = ax.plot(0, 0, '--', color='blue')
    basicline, = ax.plot(0, 0, ':', color='blue')
    # On ajoute la première légende
    leg1 = ax.legend(loc='upper right', fontsize=10)
    # On met ensuite la deuxième légende
    leg2 = ax.legend([maxline,minline, basicline],['1h','40m', '20m'], loc='right', fontsize=10)
    # Et on fait revenir la première légende (astuce pour légender deux fois)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.ylim(0, 0.3)
    plt.xlim(-1, 28)
    plt.title("Densité statistique des intervalles de distance")
    plt.xlabel("Intervalle de distance (en mètres)")
    plt.ylabel("Probabilité effective (0.4 = 40%)")
    plt.savefig("ACT-GAUSIENNE-DYNAMIQUE-DISTANCE.png", dpi=500)
    
    
    
    # Répartition des intervalles de temps en fonction des différentes activités et des capteurs
   
    AK = GAUSSIENNE2(A[2])
    BK = GAUSSIENNE2(B[2])
    CK = GAUSSIENNE2(C[2])
    DK = GAUSSIENNE2(D[2])
    EK = GAUSSIENNE2(E[2])
    FK = GAUSSIENNE2(F[2])
    GK = GAUSSIENNE2(G[2])
    HK = GAUSSIENNE2(H[2])
    IK = GAUSSIENNE2(I[2])
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(CK[0], CK[1], color="green", label="935 XT")
    ax.plot(FK[0], FK[1], 'g--')
    ax.plot(IK[0], IK[1], 'g:')
    maxline, = ax.plot(0, 0, '-', color='green')
    minline, = ax.plot(0, 0, '--', color='green')
    basicline, = ax.plot(0, 0, ':', color='green')
    # On ajoute la première légende
    leg1 = ax.legend(loc='upper right', fontsize=10)
    # On met ensuite la deuxième légende
    leg2 = ax.legend([maxline,minline, basicline],['1h','40m', '20m'], loc='right', fontsize=10)
    # Et on fait revenir la première légende (astuce pour légender deux fois)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    plt.title("Densité statistique des intervalles de temps")
    plt.xlim(-1, 7)
    plt.ylim(0, 0.5)
    plt.xlabel("Intervalle de temps (en secondes)")
    plt.ylabel("Probabilité effective (0.1 = 10%)")
    plt.savefig("ACT-GAUSIENNE-DYNAMIQUE-TEMPS-935-XT.png", dpi=500)
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # On construit chaque courbe
    ax.plot(AK[0], AK[1], color="blue", label="EDGE 1000")
    ax.plot(BK[0], BK[1], color="red", label="735 XT")
    ax.plot(CK[0], CK[1], color="green", label="935 XT")
    ax.plot(DK[0], DK[1], "b--")
    ax.plot(EK[0], EK[1], "r--")
    ax.plot(FK[0], FK[1], "g--")
    ax.plot(GK[0], GK[1], "b:")
    ax.plot(HK[0], HK[1], "r:")
    ax.plot(IK[0], IK[1], "g:")
    # On fixe les paramètres
    maxline, = ax.plot(0, 0, '-', color='blue')
    minline, = ax.plot(0, 0, '--', color='blue')
    basicline, = ax.plot(0, 0, ':', color='blue')
    # On refait l'astuce de la double légende (voir plus haut)
    leg1 = ax.legend(loc='upper right', fontsize=10)
    leg2 = ax.legend([maxline,minline, basicline],['1h','40m', '20m'], loc='right', fontsize=10)
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    # On met en forme le dernier graphique de répartition
    plt.title("Densité statistique des intervalles de temps")
    plt.xlim(-1, 7)
    plt.ylim(0, 0.5)
    plt.xlabel("Intervalle de temps (en secondes)")
    plt.ylabel("Probabilité effective (0.1 = 10%)")
    plt.savefig("ACT-GAUSIENNE-DYNAMIQUE-TEMPS.png", dpi=500)
    
    
    # Ensuite, on vient représenter les écarts absolus des activités enregistrées par les différents capteurs
    
    # Activité dynamique (40 minutes) et 50 tours
    plt.clf()
    plt.scatter(F[0],F[4], color='green', label="Garmin 935 XT \n50 tours de 253,5 m en 40 minutes", s=1)
    plt.title("Écart absolu entre la mesure GPS et la distance réelle parcourue")
    plt.subplots_adjust(top=0.80)
    
    # Données du compteur mécanique (calcul fait à la fin de l'expérience le jour-même)
    VitesseReelle = [5.25497512438, 5.1399026764, 5.2390465693]
    VitesseMecaniqueCalculee = [5.30679934, 5.19059205, 5.30365485]
    
    plt.legend(loc=2, fontsize=10)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("TIPE-Ecart absolu expérience dynamique (format 40min 50 tours)-735-XT.png", dpi=500)
    
    
    
    # Activité dynamique (1 heure) et 75 tours
    plt.clf()
    plt.scatter(A[0],A[4], color='blue', label="Garmin Edge 1000", s=1, marker="+")
    plt.scatter(B[0],B[4], color='red', label="Garmin 735 XT", s=1, marker="+")
    plt.scatter(C[0],C[4], color='green', label="Garmin 935 XT \n75 tours de 253,5 m en 60 minutes", s=1, marker="+")
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 1h
    x = np.linspace(0, 3629, 4000)
    y = []
    b = abs((VitesseReelle[2]*3000)-(VitesseMecaniqueCalculee[2]*3000))
    for i in x:
        y.append(b)
    plt.plot(x,y,color="black", label="Compteur Mécanique")
    plt.subplots_adjust(top=0.80)
    plt.legend(loc=2, fontsize=10)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("TIPE-Ecart absolu expérience dynamique (format 1h 75 tours).png", dpi=500)
    
    
    
    # Activité dynamique (40 minutes) et 50 tours
    plt.clf()
    plt.scatter(D[0],D[4], color='blue', label="Garmin Edge 1000", s=1)
    plt.scatter(E[0],E[4], color='red', label="Garmin 735 XT", s=1)
    plt.scatter(F[0],F[4], color='green', label="Garmin 935 XT \n50 tours de 253,5 m en 40 minutes", s=1)
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 40 min
    x1 = np.linspace(0, 2400, 3000)
    y1=[]
    c = abs((VitesseReelle[1]*3000)-(VitesseMecaniqueCalculee[1]*3000))
    for i in x1 :
        y1.append(c)
    plt.plot(x1, y1, color="black", alpha=0.6)
    plt.title("Écart absolu entre la mesure GPS et la distance réelle parcourue")
    plt.subplots_adjust(top=0.80)
    plt.legend(loc=2, fontsize=10)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("TIPE-Ecart absolu expérience dynamique (format 40min 50 tours).png", dpi=500)
    
    
    # Activité dynamique (20 minutes) et 25 tours
    plt.clf()
    plt.scatter(G[0],G[4], color='blue', label="Garmin Edge 1000", s=1)
    plt.scatter(H[0],H[4], color='red', label="Garmin 735 XT", s=1)
    plt.scatter(I[0],I[4], color='green', label="Garmin 935 XT \n25 tours de 253,5 m en 20 minutes", s=1)
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 20 min
    x2 = np.linspace(0, 1200, 2000)
    y2=[]
    d = abs(130)
    for i in x2 :
        y2.append(d)
    plt.plot(x2, y2, color="black", alpha=0.4)
    plt.subplots_adjust(top=0.80)
    plt.legend(loc=2, fontsize=10)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("TIPE-Ecart absolu expérience dynamique (format 20min 25 tours).png", dpi=500)
    
    
    
    # Superposition des activités dynamiques en ajoutant la comparaison aux capteurs mécaniques
    
    # Données du compteur mécanique (calcul fait à la fin de l'expérience le jour-même)
    VitesseReelle = [5.25497512438, 5.1399026764, 5.2390465693]
    VitesseMecaniqueCalculee = [5.30679934, 5.19059205, 5.30365485]
    
    # Reset du cadre graphique
    plt.clf()
    
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 1h
    x = np.linspace(1450, 3629, 4000)
    y = []
    b = abs((VitesseReelle[2]*3000)-(VitesseMecaniqueCalculee[2]*3000))
    for i in x:
        y.append(b)
    plt.plot(x,y,color="black", label="Compteur Mécanique")
    
    
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 40 min
    x1 = np.linspace(0, 2400, 3000)
    y1=[]
    c = abs((VitesseReelle[1]*3000)-(VitesseMecaniqueCalculee[1]*3000))
    for i in x1 :
        y1.append(c)
    plt.plot(x1, y1, color="black", alpha=0.6)
    
    
    # Droite représentant l'écart absolu entre la réalité et 
    # les données établis par le capteur mécanique sur le format 20 min
    x2 = np.linspace(0, 1200, 2000)
    y2=[]
    d = abs(130)
    for i in x2 :
        y2.append(d)
    plt.plot(x2, y2, color="black", alpha=0.4)
    
    
    # Représentation des écarts globaux sur toutes les activités
    plt.scatter(A[0],A[4], color='blue', label="Garmin Edge 1000", s=1, marker="+",)
    plt.scatter(B[0],B[4], color='red', label="Garmin 735 XT", s=1, marker="+")
    plt.scatter(C[0],C[4], color='green', label="Garmin 935 XT", s=1, marker="+")
    plt.scatter(D[0],D[4], color='blue', s=1, marker="+", alpha=0.3)
    plt.scatter(E[0],E[4], color='red', s=1, marker="+", alpha=0.3)
    plt.scatter(F[0],F[4], color='green', s=1, marker="+", alpha=0.3)
    plt.scatter(G[0],G[4], color='blue', s=1, marker="+", alpha=0.15)
    plt.scatter(H[0],H[4], color='red',  s=1, marker="+", alpha=0.15)
    plt.scatter(I[0],I[4], color='green', s=1, marker="+", alpha=0.15)
    plt.title("Écart absolu entre la mesure GPS et la distance réelle parcourue")
    plt.subplots_adjust(top=0.80)
    plt.legend(loc="upper left", fontsize=9)
    plt.ylim(-5, 250)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("Ecart mesuré total.png", dpi=800)
    plt.clf()
    
    #faire ressortir les derniers points : `
    plt.plot(x,y,color="black", label="Compteur Mécanique")
    plt.plot(x1, y1, color="black", alpha=0.6)
    plt.plot(x2, y2, color="black", alpha=0.4)
    # Représentation des écarts globaux sur toutes les activités
    plt.scatter(A[0],A[4], color='blue', label="Garmin Edge 1000", s=1, marker="+",)
    plt.scatter(B[0],B[4], color='red', label="Garmin 735 XT", s=1, marker="+")
    plt.scatter(C[0],C[4], color='green', label="Garmin 935 XT", s=1, marker="+")
    plt.scatter(D[0],D[4], color='blue', s=1, marker="+", alpha=0.3)
    plt.scatter(E[0],E[4], color='red', s=1, marker="+", alpha=0.3)
    plt.scatter(F[0],F[4], color='green', s=1, marker="+", alpha=0.3)
    plt.scatter(G[0],G[4], color='blue', s=1, marker="+", alpha=0.15)
    plt.scatter(H[0],H[4], color='red',  s=1, marker="+", alpha=0.15)
    plt.scatter(I[0],I[4], color='green', s=1, marker="+", alpha=0.15)
    plt.title("Écart absolu entre la mesure GPS et la distance réelle parcourue")
    plt.subplots_adjust(top=0.80)
    plt.legend(loc="upper right", fontsize=7)
    plt.ylim(-5, 250)
    plt.xlim(3600,3650)
    plt.xlabel("temps (s)")
    plt.ylabel("écart mesuré (m)")
    plt.savefig("Ecart mesuré total, derniers points.png", dpi=800)
    plt.clf()
    
    # On va ressortir les trois derniers écarts absolus mesurés entre la réalité et chaque capteur GPS
    T = [A,B,C,D,E,F,G,H,I]
    print("\n\n\n**************** Trois derniers écarts mesurées en mètres ****************")
    
    # afficher les données qui nous intéressent, en les regroupant
    for i in T :
        if i==A or i==B or i==C :
            duree_activite = "1 heure"
        elif i==D or i==E or i==F :
            duree_activite = "40 minutes"
        elif i==G or i==H or i==I :
            duree_activite = "20 minutes"
        if i==A or i==D or i==G :
            capteur="Garmin Edge 1000"
        elif i==B or i==E or i==H :
            capteur="Garmin 735 XT"
        elif i==C or i==F or i==I :
            capteur="Garmin 935 XT"
            
        if i==A or i==D or i==G :
            print("\n\n-------------- Activité format : "+duree_activite+" --------------")
        print("\n \n"+capteur+" : \n")
        for j in range(-3, 0, 1):
            print(" "+str(i[4][j]))
    
    # On a obtenu nos quatre graphiques et des données extraites intéressantes, on retourne dans le reste du programme
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
    mesure_des_ecarts_dynamique()

Start()

