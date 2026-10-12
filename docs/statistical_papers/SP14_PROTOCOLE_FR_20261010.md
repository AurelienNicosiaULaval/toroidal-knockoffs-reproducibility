# Protocole SP14: moyenne représentée et dimension croissante

10 octobre 2026. Fixé avant les nouveaux calculs SP14. Cette version complète SP13 sans modifier ses expériences. Source Python exécutée; pas de portage R annoncé.

Question: dans le même modèle de paires iid, la représentation exacte de la moyenne avec H=2 permet-elle de transférer le contrôle des signes par comparaison de la loi jointe des coefficients gaussiens? Régime candidat: K,m divergent, K²/n tend vers zéro, sans condition sur m/n. Le champ est connu, seul g est appris sur K m incréments indépendants. Bruit gaussien indépendant et ridge positif fixé. Aucune extension générale aux graphes appris ou aux réponses binaires.

La note SP14_REPRESENTED_PROOF_EN_20261010.md contient l'argument à contrôler: loi gaussienne conditionnelle exacte, covariance de dimension 16K, erreur de Frobenius O_P(K/√n+1/√m), transfert de la même fonction bornée de FDP en variation totale, séparation des actifs. Le transfert en variation totale et l'inférence gaussienne sont des outils connus; aucune priorité générale n'est revendiquée.

Tests numériques fixés:

1. Moments H=2 d'une paire à g=0,2 et κ=0,04, avec paramètre de travail 0,18, 0,20, 0,22. Quadratures (24 angles,16 Hermite), puis (32,24); si écart >10^-8, raffinement (48,32), sans sélectionner une précision selon le nombre de découvertes. Contrôler positivité, cible de régression [cos 2X1 =1, reste=0], symétrie de covariance par échange nul à t=g et variation lorsque t change. La réponse-covariance est intégrée, pas seulement extraite d'une colonne déclarée de la covariance.
2. Réanalyser les neuf jeux SP12 conservés, chacun une fois avec H=2: mêmes sujets, pilotes, témoins, réponse et bruit. Aucun nouveau tirage de sujets. Conserver toutes les sorties H=1 et toutes les sélections H=2, y compris vides. Fixer q=0,1 et λ=1. Vérifier l'identité ridge conditionnelle à partir du bruit enregistré, calculer la covariance conditionnelle et une borne de comparaison gaussienne avec l'oracle connu, limitée à un diagnostic de loi conditionnelle.
3. Vérification indépendante: caractéristiques dans un ordre différent, résolution par QR augmenté sur les trois cas n=2000 (K=5,10,20), recalcul indépendant du filtre pour les neuf cas et comparaison des coefficients. Contrôler les moments von Mises H=2 contre une intégration angulaire directe. Inclure un contrôle négatif de pénalité mal normalisée et un contrôle de non-représentation H=1.

Seuils: 10^-8 pour convergence des moments et cible/symétrie; 10^-10 pour l'identité et les coefficients QR; sélections identiques. Une positivité numérique ne certifie pas un minimum uniforme théorique. Les calculs de lois gaussiennes ne sont pas des simulations toroïdales de FDR.

Les neuf réanalyses ne fournissent ni FDR Monte-Carlo ni garantie de puissance empirique générale. Rapporter FDP, puissance et vide pour chaque jeu, sans moyenne présentée comme calibration. Les champs et la loi vraie connus ne servent qu'aux diagnostics, pas au réglage du filtre.

Provenance: copie locale des neuf NPZ dont les SHA-256 sont contrôlés contre le reçu SP12. Copies par buffers pour éviter le défaut de copie File Provider constaté; reçu créé avant toute lecture longue, état incomplet conservé en cas d'erreur. Sorties versionnées avec source, protocole et empreintes des entrées. Aucune réécriture d'une ancienne sortie. Pas de soumission ni de publication.

Décision: intégrer un résultat seulement après fermeture de la preuve et contrôles exécutés; si une étape échoue, conserver l'échec et restreindre l'énoncé. L'absence de précédent identifié ne certifie pas la nouveauté.
