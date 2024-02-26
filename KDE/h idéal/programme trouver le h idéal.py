import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# générer l'échantillon à partir de deux lois normales : 
N = 10000 # mettre le nombre de points voulu

X = np.concatenate(
    (np.random.normal(0, 1, int(0.4 * N)), np.random.normal(5, 2, int(0.6 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# préparation de l'affichage de la vraie densité, qui est celle à partir de laquelle les données 
#ont été générées (voir plus haut)
#la pondération des lois dans la somme est la pondération des lois dans l'échantillon généré (voir plus haut)
true_dens = 0.4 * norm(0, 1).pdf(X_plot[:, 0]) + 0.6 * norm(5, 2).pdf(X_plot[:, 0])

bandwidth = np.arange(0.1, 1, .05) #parcours de toutes les valeurs de h de 0.05 à 2 avec un pas de 0.05 
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth}) #fait toutes les combinaisons des paramètres
grid.fit(X)

fig = plt.figure()
ax = fig.add_subplot(111) #format du graphique
kde = grid.best_estimator_ #meilleur modèle
log_dens = kde.score_samples(X_plot)
density=np.exp(log_dens)
ax.fill(X_plot[:,0], true_dens, fc='black', alpha=0.2, label='distribution en entrée')
ax.plot(X_plot[:,0], density, '-', label="Estimation")
ax.legend(loc='upper left')
ax.set_title("Estimation optimal avec le noyau Gaussien")

#Ajout de la légende : 
ax.set_xlabel("x")
ax.set_ylabel("Densité normalisée")

#affichage du nombre de points de la mesure et de la valeur optimale de h : 
ax.text(4, 0.28, "N={0} points".format(N))
ax.text(4, 0.25, "optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))

#limites du graphique : 
ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.3)

#sauvegarde du résultat 
plt.savefig("Estimation optimal avec le noyau gaussien, N=10000.png", dpi=500)
