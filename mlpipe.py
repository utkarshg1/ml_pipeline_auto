from pathlib import Path
from typing import Literal
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import cross_val_score


class MLWorkFlow:
    def __init__(
        self,
        path: Path,
        problem: Literal["classification", "regression"],
        drop: list[str],
        target: str,
        test_size: float = 0.2,
    ):
        self.path = path
        self.data = pd.read_csv(path)
        self.problem = problem
        self.drop = drop
        self.target = target
        self.test_size = test_size
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.model = None
        self.results = None

    def preprocess_data(self):
        """Preprocess the data by dropping unnecessary columns and splitting into train and test sets."""
        # Drop unnecessary columns
        self.data.drop(columns=self.drop, inplace=True)
        # Split into features and target
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target].values.flatten()
        # Preprocessing
        categorical_features = self.X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_features = self.X.select_dtypes(include="number").columns.tolist()
        # Define the preprocessing steps
        num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numerical_features),
            ]
        )
        if categorical_features:
            cat_pipe = make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
            )
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipe, numerical_features),
                    ("cat", cat_pipe, categorical_features),
                ]
            )
        X_pre = self.preprocessor.fit_transform(X)
        # Save the preprocessor to a file
        joblib.dump(self.preprocessor, "preprocessor.joblib")
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_pre, y, test_size=self.test_size, random_state=42
        )

    def get_models(self):
        if self.problem == "classification":
            return [
                LogisticRegression(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                GradientBoostingClassifier(),
                XGBClassifier(),
            ]
        elif self.problem == "regression":
            return [
                LinearRegression(),
                Ridge(),
                Lasso(),
                DecisionTreeRegressor(),
                RandomForestRegressor(),
                GradientBoostingRegressor(),
                XGBRegressor(),
            ]
        else:
            raise ValueError(
                "Problem type must be either 'classification' or 'regression'"
            )

    def evaluate_model(self, model):
        # Fit model
        model.fit(self.X_train, self.y_train)
        # Predict results on train and test sets
        ypred_train = model.predict(self.X_train)
        ypred_test = model.predict(self.X_test)
        # Evaluate model
        if self.problem == "classification":
            train_score = f1_score(self.y_train, ypred_train, average="macro")
            test_score = f1_score(self.y_test, ypred_test, average="macro")
            cv_score = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring="f1_macro", n_jobs=-1
            ).mean()
        elif self.problem == "regression":
            train_score = r2_score(self.y_train, ypred_train)
            test_score = r2_score(self.y_test, ypred_test)
            cv_score = cross_val_score(
                model, self.X_train, self.y_train, cv=5, scoring="r2", n_jobs=-1
            ).mean()

        # Model name
        name = model.__class__.__name__
        return {
            "name": name,
            "model": model,
            "train_score": train_score,
            "test_score": test_score,
            "cv_score": cv_score,
        }

    def evaluate_all_models(self):
        """Evaluate all models and return the best one."""
        models = self.get_models()
        results = []
        for model in models:
            result = self.evaluate_model(model)
            results.append(result)
        # Sort results by test score
        results.sort(key=lambda x: x["test_score"], reverse=True)
        self.results = pd.DataFrame(results)
        self.model = self.results.loc[0, "model"]
        # Dump model to file
        joblib.dump(self.model, "model.joblib")
        return self.results, self.model

    def run(self):
        self.preprocess_data()
        results, model = self.evaluate_all_models()
        print("Best model: ", results.loc[0, "name"])
        print(results)
