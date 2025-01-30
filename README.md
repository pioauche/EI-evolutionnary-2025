# ST2 (Théorie des Jeux) - EI Algorithmique Génétique

# Description
Ce projet implémente deux algorithmes d'optimisation : le recuit simulé et un algorithme génétique, 
pour ajuster les rotations de nucléotides afin de rendre une séquence d’ADN circulaire.

# Installation
Pour installer les dépendances (qui sont seulement numpy et matplotlib), exécutez la commande suivante : <code>pip install -r requirements.txt</code>

# Utilisation
Pour lancer le programme, il vous suffit d'exécuter dans la console la commande : <code>python -m 3dna data/plasmid_8k.fasta --genetic --annealing --generations 200 --population 100 --mutation_rate 0.1</code>
Vous pouvez en apprendre plus sur les paramètres avec : <code>python -m 3dna --help</code>

Les paramètres --genetic et --annealing vous permettent de choisir l'algorithme d'optimisation utilisé.
--generations et --population servent à l'algorithme génétiques. Ils ont comme valeurs par défaut 100 et 50.
--mutation_rate représente la probabilité de mutation de chaque gène.

Voici ce que l'on pourrait obtenir en exécutant le programme :
![Exemple de graphe](data/example.png)
