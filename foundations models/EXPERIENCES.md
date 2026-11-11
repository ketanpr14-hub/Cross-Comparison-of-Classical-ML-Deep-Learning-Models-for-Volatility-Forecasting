# Prévision de la volatilité des actions

On prévoit log(rv5) à J+1 sur les 40 actions VOLARE.
On mesure l’erreur avec la MSE et on prend le HAR de base de Quentin comme référence.

1. **Test de départ.** On teste Chronos-2, Timestral-209M, TimesFM 3.0 et TTM-R3 avec 22 puis 1 024 séances d’historique, sur les mêmes dates. À 22 séances, HAR est devant les quatre modèles. À 1 024 séances, les quatre passent devant HAR.
2. **Variantes optimisées.** On choisit la profondeur et les réglages sur 2020, puis on évalue à partir de 2021. Le meilleur résultat est Chronos adapté avec LoRA, mélangé au HAR enrichi (25 % Chronos, 75 % HAR). Sa MSE est de 0,27098, contre 0,27250 pour le HAR enrichi seul entraîné sur la même période.
3. **Fine-tuning sans HAR.** On compare chaque modèle avant et après adaptation, à profondeur et dates identiques. Chronos utilise LoRA. Pour TTM, Timestral et TimesFM, on entraîne seulement la tête de prévision. La MSE baisse de 0,99 % pour TTM, 0,15 % pour TimesFM et 0,09 % pour Timestral. Elle augmente de 0,56 % pour Chronos.
