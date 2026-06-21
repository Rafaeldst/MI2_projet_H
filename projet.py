import numpy as np

# Choix des variables
hbar= 1.0
m=1
k0=5
x_min, x_max = -20.0, 20.0
t_min, t_max = 0.0, 4.0
nx = 500
nt = 8000
V0=17.5
largeur_a=1
a0=1

x_tab = np.linspace(x_min, x_max, nx)
dx = x_tab[1] - x_tab[0]
t_tab = np.linspace(t_min, t_max, nt)
dt = t_tab[1] - t_tab[0]

T_entree=0
T_sortie=0

Psi = np.zeros((nx, nt), dtype=complex)

def GaussWP(k0, a, x, t_val):#Créer le paquet gaussien
    prefacteur = (1 / (8 * np.pi**3))**0.25
    denominateur = m * a**2 + 2j * hbar * t_val
    racine = np.sqrt((4 * np.pi * m * a) / denominateur)
    terme_exp_1 = (m / 4) * ((a**2 * k0 + 2j * x)**2 / denominateur)
    terme_exp_2 = (a**2 * k0**2) / 4
    return prefacteur * racine * np.exp(terme_exp_1 - terme_exp_2)

#Création de la barrière
V=np.zeros(nx)
for i in range (nx):
    if (x_tab[i]>=a0 and x_tab[i]<=a0+largeur_a):
        V[i]=V0

Psi[:, 0] = GaussWP(k0, 1.0, x_tab, t_min)

#euler pour initialiser t=1
d2_spatiale = (Psi[2:, 0] - 2 * Psi[1:-1, 0] + Psi[:-2, 0]) / (dx**2)
d_temporelle = (1j * hbar / (2 * m)) * d2_spatiale - (1j / hbar) * V[1:-1] * Psi[1:-1, 0]
Psi[1:-1, 1] = dt * d_temporelle + Psi[1:-1, 0]

for t in range(1, nt - 1):
    d2_spatiale = (Psi[2:, t] - 2 * Psi[1:-1, t] + Psi[:-2, t]) / (dx**2)
    d_temporelle = (1j * hbar / (2 * m)) * d2_spatiale - (1j / hbar) * V[1:-1] * Psi[1:-1, t]
    Psi[1:-1, t+1] = Psi[1:-1, t-1] + 2 * dt * d_temporelle

#indice du mur
indice_entree = np.argmin(np.abs(x_tab - a0))
indice_sortie = np.argmin(np.abs(x_tab - (a0 + largeur_a)))

#récupère les probabilités de présence à partir de l'entrée et de la sortie
proba_entree_temps = np.abs(Psi[indice_entree, :])**2
proba_sortie_temps = np.abs(Psi[indice_sortie, :])**2

#récupère l'indice plus grande probabilité de présence
t_entree_idx = np.argmax(proba_entree_temps)
T_entree = t_tab[t_entree_idx]

#récupère le temps d'entrée et de sortie
t_sortie_idx = np.argmax(proba_sortie_temps)
T_sortie = t_tab[t_sortie_idx]

Tnum =T_sortie - T_entree

if V0!=0:
    print("==========================================")
    print(f"Hauteur du mur (V0)  = {V0}")
    print(f"Épaisseur du mur (a) = {largeur_a:.1f}")
    print(f"Entrée  (x = {a0:.1f}) à t = {T_entree:.4f}")
    print(f"Sortie  (x = {a0 + largeur_a:.1f}) à t = {T_sortie:.4f}")
    print(f"--> Temps de traversée (tau_num) = {Tnum:.4f}")
    print("==========================================\n")

else:
    v_groupe = hbar * k0 / m
    T0 = largeur_a / v_groupe
    erreur_rel = abs(Tnum - T0) / T0
    print("==========================================")
    print(f"Hauteur du mur (V0)  = {V0}")
    print(f"Épaisseur du mur (a) = {largeur_a:.1f}")
    print(f"Vitesse de groupe    = {v_groupe:.2f}")
    print(f"τ théorique (a/v_g)  = {T0:.4f} s")
    print(f"Entrée  (x = {a0:.1f}) à t = {T_entree:.4f}")
    print(f"Sortie  (x = {a0 + largeur_a:.1f}) à t = {T_sortie:.4f}")
    print(f"--> Temps de traversée (T_num) = {Tnum:.4f}")
    print(f"--> Écart relatif = {erreur_rel:.4f}")
    print("==========================================\n")
