# Prévision de la volatilité des actions

On prévoit `log(rv5)` à J+1 sur les 40 actions VOLARE. On compare les prévisions au HAR de base de Quentin avec la MSE sur le log et QLIKE sur la variance. La correction `sigma2` est estimée sur les erreurs de validation.

1. **Test de départ.** On teste les quatre modèles avec 22 puis jusqu’à 1 024 séances. Le contexte raccourci au début de 2019 permet de retrouver les 78 605 prévisions de Quentin. Le CSV garde aussi les scores sur les 77 925 dates à contexte complet pour cette coupure.
2. **Variantes optimisées.** On choisit les réglages sur 2018 ou 2020, avant chaque coupure. Les meilleurs QLIKE sont ceux du mélange TimesFM–HAR en 2019 (0,18242) et du mélange Chronos–HAR en 2021 (0,17631).
3. **Fine-tuning sans HAR.** On compare chaque modèle avant et après adaptation, à profondeur et dates identiques pour chaque coupure. Chronos utilise LoRA. Pour TTM, Timestral et TimesFM, on entraîne la tête de prévision.

Tous les scores sont dans le [CSV de résultats](results/stocks_foundation_models.csv).
