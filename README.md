# Volatility_Prediction
Repository which contains a part of my project with Amundi, on the prediction of historical volatility with GRU/LSTM networks

# Structure du Repository GitHub - Volatility Forecasting

## 📁 Structure recommandée du repository

```
volatility-forecasting-garch/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/
│   │   └── data_macro_romain_processed.csv
│   └── processed/
│       └── (données nettoyées si nécessaire)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_garch_modeling.ipynb
│   └── 03_volatility_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── garch_models.py
│   ├── forecasting.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_forecasting.py
├── results/
│   ├── figures/
│   └── model_outputs/
├── config/
│   └── config.yaml
└── docs/
    ├── methodology.md
    └── results_interpretation.md
```

## 📝 Contenu des fichiers principaux

### README.md
```markdown
# Volatility Forecasting with GARCH Models

## 🎯 Objectif
Modélisation et prévision de la volatilité du taux de change EUR/USD en utilisant des modèles GARCH avec ré-estimation périodique des paramètres.

## 📊 Données
- Série temporelle des rendements EUR/USD
- Période d'étude : [spécifier la période]
- Fréquence : quotidienne

## 🔧 Modèles utilisés
- GARCH(1,1)
- EGARCH(1,1)
- GJR-GARCH(1,1)
- Distributions : Normal, Student-t, GED

## 📈 Méthodologie
1. **Analyse exploratoire** des données
2. **Sélection du modèle optimal** (AIC/BIC)
3. **Prévision avec ré-estimation périodique**
4. **Évaluation des performances** (MSE, MAE, Theil)

## 🚀 Installation et utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

### Utilisation
```bash
# 1. Exploration des données
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Modélisation GARCH
jupyter notebook notebooks/02_garch_modeling.ipynb

# 3. Prévision de volatilité
jupyter notebook notebooks/03_volatility_forecasting.ipynb
```

## 📊 Résultats principaux
- Modèle optimal : GARCH(1,1) avec distribution Student-t
- Performance de prévision : [ajouter vos métriques]
- Impact de la ré-estimation : amélioration significative avec ré-estimation quotidienne

## 📁 Structure du projet
- `data/` : Données brutes et traitées
- `notebooks/` : Notebooks Jupyter d'analyse
- `src/` : Code source modulaire
- `results/` : Graphiques et résultats
- `tests/` : Tests unitaires

## 📚 Références
- [Lahmiri et al.] : Méthodologie de base
- [Autres références académiques]

## 📄 License
MIT License
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
yfinance>=0.1.70
arch>=5.3.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
jupyter>=1.0.0
seaborn>=0.11.0
plotly>=5.0.0
```

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Jupyter Notebook
.ipynb_checkpoints

# Data files (si trop volumineux)
data/raw/*.csv
!data/raw/sample_data.csv

# Results
results/temp/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 🔧 Refactoring du code

### Étape 1: Diviser le notebook en modules

**src/garch_models.py**
```python
"""
Modèles GARCH pour la prévision de volatilité
"""
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GARCHForecaster:
    def __init__(self, model_type='Garch', distribution='StudentsT', p=1, q=1):
        self.model_type = model_type
        self.distribution = distribution
        self.p = p
        self.q = q
        self.model_result = None
    
    def fit(self, returns):
        """Ajuster le modèle GARCH"""
        # Votre logique de fit_model() ici
        pass
    
    def forecast_with_reestimation(self, train_data, test_data, reestimation_period=1):
        """Prévision avec ré-estimation périodique"""
        # Votre logique de forecasting ici
        pass
```

### Étape 2: Organiser les notebooks

**01_data_exploration.ipynb**
- Chargement et nettoyage des données
- Statistiques descriptives
- Tests de stationnarité
- Visualisations exploratoires

**02_garch_modeling.ipynb**
- Sélection du modèle optimal
- Comparaison des distributions
- Analyse des résidus
- Validation du modèle

**03_volatility_forecasting.ipynb**
- Prévisions hors-échantillon
- Impact de la ré-estimation
- Évaluation des performances
- Visualisations des résultats

## 📊 Améliorations suggérées

### 1. Documentation
- Docstrings détaillées pour toutes les fonctions
- Commentaires explicatifs dans le code
- Méthodologie clairement expliquée

### 2. Visualisations professionnelles
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_professional_plot():
    fig, ax = plt.subplots(figsize=(12, 8))
    # Votre code de visualisation
    plt.tight_layout()
    plt.savefig('results/figures/volatility_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 3. Configuration centralisée
**config/config.yaml**
```yaml
data:
  file_path: "data/raw/data_macro_romain_processed.csv"
  return_column: "return_eurousd"
  split_ratio: 0.8

models:
  garch:
    p: 1
    q: 1
    distributions: ["Normal", "StudentsT", "ged"]
  
forecasting:
  reestimation_periods: [1, 5, 20, 60]
  horizon: 1

evaluation:
  metrics: ["MSE", "MAE", "Theil"]
```

### 4. Tests unitaires
**tests/test_models.py**
```python
import unittest
import numpy as np
import pandas as pd
from src.garch_models import GARCHForecaster

class TestGARCHModels(unittest.TestCase):
    def setUp(self):
        # Données de test
        self.test_returns = pd.Series(np.random.normal(0, 0.01, 1000))
    
    def test_garch_fitting(self):
        """Test d'ajustement du modèle GARCH"""
        forecaster = GARCHForecaster()
        forecaster.fit(self.test_returns)
        self.assertIsNotNone(forecaster.model_result)

if __name__ == '__main__':
    unittest.main()
```

## 🚀 Commandes pour créer le repository

```bash
# 1. Initialiser le repository
git init volatility-forecasting-garch
cd volatility-forecasting-garch

# 2. Créer la structure
mkdir -p {data/raw,data/processed,notebooks,src,tests,results/figures,results/model_outputs,config,docs}

# 3. Créer les fichiers principaux
touch README.md requirements.txt .gitignore LICENSE
touch src/__init__.py tests/__init__.py

# 4. Premier commit
git add .
git commit -m "Initial project structure"

# 5. Ajouter remote et push
git remote add origin https://github.com/username/volatility-forecasting-garch.git
git push -u origin main
```

## 💡 Conseils supplémentaires

1. **Badge GitHub** : Ajoutez des badges pour la licence, Python version, etc.
2. **Actions GitHub** : Configurez CI/CD pour les tests automatiques
3. **Documentation** : Considérez Sphinx pour une documentation plus avancée
4. **Reproductibilité** : Fixez les seeds pour les résultats reproductibles
5. **Performance** : Ajoutez des benchmarks de temps d'exécution
