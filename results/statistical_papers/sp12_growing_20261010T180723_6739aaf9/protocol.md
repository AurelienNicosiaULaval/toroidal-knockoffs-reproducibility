# Protocole SP12, fixé avant les calculs

Question: réaliser l'obstruction du pilote partagé avec un nombre croissant de signaux, sans attribuer une garantie positive de grande dimension au projet. Ce plan succède à SP11 et ne modifie aucune ancienne configuration.

## Porte analytique

Modèle: K paires indépendantes avec interaction commune g>0 et champ connu kappa=g^2 suffisamment petit pour le lemme à une paire. La réponse est la somme des seconds harmoniques des premiers angles, plus un bruit gaussien indépendant de variance un. Tous les K premiers angles sont non nuls; leurs compagnons sont conditionnellement nuls. Le nombre de signaux croît, mais leur proportion est un demi. Ce n'est pas un régime sparse et p/n doit tendre vers zéro dans la route examinée.

Le MLE du pilote utilise Km incréments iid, donc la preuve doit dériver son échelle sqrt(Km). La base est H=1 et la pénalité ridge vaut un. Démontrer uniformément près de g: une covariance de population à blocs indépendants, un minorant spectral constant, une erreur de covariance empirique en norme d'opérateur, puis une erreur euclidienne de régression O_P(K/sqrt(n)). Examiner la condition suffisante n/(K^3 m) tendant vers l'infini, son rôle dans les signes nuls communs et le seuil knockoff+ réel. Ne pas substituer un taux à dimension fixe.

Contrôle supplémentaire: déterminer l'échelle marginale des scores oracle et si un couplage quelconque pourrait approcher les scores ajustés à cette échelle. Comparer précisément aux conditions d'ARK, dans sa version arXiv du 4 juin 2024, et de Fan et al., dans la copie auteur du 12 juin 2026 conservée. Une preuve négative ne doit pas être formulée comme une contradiction de leurs théorèmes.

Critère de poursuite: une preuve uniforme complète avec traitement du centrage, du terme ridge et des frontières de signe. Si un terme ne converge pas, conserver l'obstacle et restreindre ou abandonner l'énoncé.

## Diagnostic numérique borné

Neuf jeux iid indépendants, K dans {5,10,20}, n dans {2000,8000,32000}, m=40, g=0.2, kappa=0.04. Graine racine 2026101201; huit flux indépendants par cas pour racines, incréments, pilote, augmentation, témoins et réponse. Aucun choix fondé sur Y, aucun retrait de graine et aucun ajustement après observation des résultats. Ce choix numérique n'est pas une certification de la prémisse analytique « suffisamment petit ».

Originaux exacts: racine von Mises(kappa) et incrément indépendant von Mises(g). Témoins: champ commun t(v1+v2)+sqrt(t)Z, puis champs individuel h2=kappa e1 et deux von Mises indépendantes. Estimer t uniquement sur le pilote; conserver toute troncature à zéro et tout échec. Stocker les angles originaux, pilotes et témoins, les normales latentes, le bruit de réponse et les coefficients.

Vérifier les coefficients de population par quadrature périodique racine/incrément et Gauss-Hermite: comparer 24^2×16^2 et 32^2×24^2, puis 48^2×32^2 si la différence dépasse 1e-9. Ne pas relâcher la tolérance. Documenter la réflexion qui annule les coefficients sinus et conserver toute insuffisance de résolution. Calculer aussi les variances résiduelles et le coefficient de variance oracle à K croissant.

Mesurer l'erreur euclidienne normalisée par K/sqrt(n), l'erreur maximale des scores autour du score de population, le minorant spectral empirique et la borne lipschitzienne déterministe. Exécuter le seuil au niveau 0.1 et conserver ses neuf FDP individuels; ne pas en tirer une estimation calibrée de FDR. Ces tailles ne sont pas supposées être dans le régime asymptotique de domination du pilote.

Un second programme doit recalculer les neuf régressions par QR, avec un ordre global des coordonnées différent, à partir des données sauvegardées; vérifier coefficients et sélections à 1e-10. Il ne doit importer ni le générateur, ni le solveur, ni le sélecteur du premier. Les données sont neuf jeux, pas neuf fois leur nombre de lignes comme répétitions indépendantes de sélection. Total prévu: 126000 sujets de sélection et 360 sujets pilotes.

## Versions et livrables

Nouveaux scripts SP12, répertoire de résultats distinct avec reçu, graines, configuration et empreintes, note de preuve et décision française. Aucune nouvelle grande étude de sélection tant que la porte analytique n'est pas franchie. Les sources SP11 et leurs déclarations doivent être conservées avant édition; une lecture déchargée par File Provider n'est pas une inspection de contenu. Les conclusions numériques défavorables restent dans le rapport. Aucun téléversement ni dépôt.
