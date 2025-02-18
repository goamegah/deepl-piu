import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import joblib

class CorrelationThreshold(BaseEstimator, TransformerMixin):
    """ Supprime les features ayant une forte corrélation (> threshold) """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.selected_features = None

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        self.selected_features = [col for col in range(X.shape[1]) if col not in to_drop]
        return self

    def transform(self, X):
        return X[:, self.selected_features]


class DataPreprocessor:
    def __init__(self, target_column='sii', feature_selection_method=None, k_best=None, 
                 imputation_method='median', drop_missing_target=True, correlation_threshold=0.9, 
                 balance_strategy=None):
        self.target_column = target_column
        self.feature_selection_method = feature_selection_method
        self.k_best = k_best
        self.imputation_method = imputation_method
        self.drop_missing_target = drop_missing_target
        self.correlation_threshold = correlation_threshold
        self.balance_strategy = balance_strategy
        self.num_features = []
        self.cat_features = []
        self.selected_features_ = None
        self.pipeline = None
        self.selector = None
        self.num_classes = None
        self.num_imputer = None
        self.cat_imputer = None

    def _validate_and_fix(self, X):
        if np.isnan(X).sum() > 0:
            print(f"🚨 NaN détectés, remplacement par 0")
            X = np.nan_to_num(X, nan=0.0)

        if np.isinf(X).sum() > 0:
            print(f"🚨 Valeurs infinies détectées, correction en cours...")
            X = np.clip(X, -1e6, 1e6)

        return X

    def fit_transform(self, df):
        if self.drop_missing_target:
            df = df.dropna(subset=[self.target_column])

        self.df_train = df.copy()
        self.identify_features()
        self.feature_engineering()

        X = self.df_train.drop(columns=[self.target_column]) if self.target_column in self.df_train.columns else self.df_train
        y = self.df_train[self.target_column] if self.target_column in self.df_train.columns else None

        self.train_features = X.columns

        if y is not None:
            self.num_classes = len(np.unique(y))

        X[self.num_features] = self.num_imputer.fit_transform(X[self.num_features])
        X[self.cat_features] = self.cat_imputer.fit_transform(X[self.cat_features])

        X_transformed = self.pipeline.fit_transform(X)

        if self.feature_selection_method:
            X_transformed = self.feature_selection(X_transformed, y)

        X_transformed, y = self.handle_class_imbalance(X_transformed, y)
        X_transformed = self._validate_and_fix(X_transformed)

        class_weights = self.compute_class_weights(y) if self.balance_strategy == 'class_weight' else None

        return torch.tensor(X_transformed, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long) if y is not None else None, class_weights

    def transform(self, df):
        """ Transforme les nouvelles données avec le pipeline entraîné """
        if not hasattr(self.pipeline, 'transformers_'):
            raise ValueError("🚨 Le pipeline de transformation n'est pas entraîné ! Appelez `fit_transform()` d'abord.")

        # ✅ Aligner les colonnes du test avec celles du train
        missing_cols = set(self.train_features) - set(df.columns)
        extra_cols = set(df.columns) - set(self.train_features)

        # Ajouter les colonnes manquantes (avec valeurs nulles ou 0 pour les catégoriques)
        for col in missing_cols:
            df[col] = 0 if col in self.cat_features else np.nan

        # Supprimer les colonnes en trop
        df = df.drop(columns=extra_cols, errors='ignore')

        # Réordonner les colonnes pour correspondre exactement à celles du train
        df = df[self.train_features]

        # ✅ Appliquer les imputeurs sauvegardés
        df[self.num_features] = self.num_imputer.transform(df[self.num_features])
        df[self.cat_features] = self.cat_imputer.transform(df[self.cat_features])

        # ✅ Appliquer la transformation principale
        X_transformed = self.pipeline.transform(df)

        # ✅ Appliquer la sélection de features sauvegardée
        if self.selector:
            X_transformed = self.selector.transform(X_transformed)
        if self.selected_features_ is not None:
            X_transformed = X_transformed[:, self.selected_features_]

        return torch.tensor(X_transformed, dtype=torch.float32)
        
    def feature_selection(self, X, y):
        """ 
        Sélection des features si activée 
        types : 'k_best', 'f_classif', 'chi2', 'lasso', 'random_forest', 'pca', 'variance_threshold', 'correlation_threshold'
        """

        
        if self.feature_selection_method == 'lasso' and y is not None:
            lasso = Lasso(alpha=0.01)
            lasso.fit(X, y)
            self.selected_features_ = np.where(lasso.coef_ != 0)[0]
            return X[:, self.selected_features_]

        elif self.feature_selection_method == 'random_forest' and y is not None:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            self.selected_features_ = np.argsort(rf.feature_importances_)[-self.k_best:]
            return X[:, self.selected_features_]
        elif self.feature_selection_method == 'logistic_regression' and y is not None:
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(X, y)
            self.selected_features_ = np.argsort(np.abs(logreg.coef_))[0][-self.k_best:]
            return X[:, self.selected_features_]
        elif self.feature_selection_method in ['k_best', 'f_classif', 'chi2', 'pca', 'variance_threshold', 'correlation_threshold']:
            self.selector = {
                'k_best': SelectKBest(score_func=mutual_info_classif, k=self.k_best),
                'f_classif': SelectKBest(score_func=f_classif, k=self.k_best),
                'chi2': SelectKBest(score_func=chi2, k=self.k_best),
                'pca': PCA(n_components=self.k_best),
                'variance_threshold': VarianceThreshold(threshold=0.01),
                'correlation_threshold': CorrelationThreshold(threshold=self.correlation_threshold),
            }[self.feature_selection_method]

        return self.selector.fit_transform(X, y) if self.selector else X

    def identify_features(self):
        """ Identifie les variables numériques et catégoriques """
        self.num_features = self.df_train.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_features = self.df_train.select_dtypes(exclude=[np.number]).columns.tolist()

        if self.target_column in self.num_features:
            self.num_features.remove(self.target_column)
        if self.target_column in self.cat_features:
            self.cat_features.remove(self.target_column)
    
    def handle_missing_values(self):
        """ Définition de l'imputation des valeurs manquantes """
        if self.imputation_method == 'median':
            num_imputer = SimpleImputer(strategy='median')
        elif self.imputation_method == 'mean':
            num_imputer = SimpleImputer(strategy='mean')
        elif self.imputation_method == 'knn':
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError("Méthode d'imputation non supportée")
    
        cat_imputer = SimpleImputer(strategy='most_frequent')
    
        # ✅ Sauvegarde des imputeurs pour une utilisation en inférence
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
    
        return num_imputer, cat_imputer

    def feature_engineering(self):
        """ Création du pipeline de preprocessing """
        num_imputer, cat_imputer = self.handle_missing_values()

        num_pipeline = Pipeline([
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', cat_imputer),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, self.num_features),
            ('cat', cat_pipeline, self.cat_features)
        ])

    def handle_class_imbalance(self, X, y):
        if self.balance_strategy == 'smote':
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X, y = smote.fit_resample(X, y)
        elif self.balance_strategy == 'random_over':
            over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            X, y = over_sampler.fit_resample(X, y)
        elif self.balance_strategy == 'random_under':
            under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X, y = under_sampler.fit_resample(X, y)
        return X, y
    

    def compute_class_weights(self, y):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        return torch.tensor(class_weights, dtype=torch.float32)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
