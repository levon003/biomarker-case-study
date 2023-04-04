import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.decomposition
import sklearn.ensemble
import sklearn.metrics
import sklearn.preprocessing
from pandas.api.types import CategoricalDtype


@dataclass
class ModelingConfig:
    """Dataclass to manage experimental configurations.
    In a larger application, could be used as an interface to an experiment tracker like MLFlow or used to load/store JSON configs.
    """

    experiment_name: str = "default"
    patient_feature_action: str = "keep"
    biomarker_feature_action: str = "keep"
    biomarker_svd_n_components: Optional[int] = None
    is_disease_sub_type_ordered: bool = True
    excluded_columns: list[str] = field(
        default_factory=lambda: [
            "status_alcohol_usage",
            "status_exercise_frequency",
            "status_bmi_level",
            "status_days_since_diagnosis",
        ],
    )
    cv_column: str = "institution_name"
    target_column: str = "target_label"

    # see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
    # these values are the HistGradientBoostingClassifier defaults
    gbm_learning_rate: float = 0.1
    gbm_max_iter: int = 100
    gbm_max_leaf_nodes: int = 31
    gbm_min_samples_leaf: int = 20
    gbm_l2_regularization: float = 0.0


class ModelEvaluator:
    def __init__(self, config: ModelingConfig, df: pd.DataFrame):
        self.config = config
        self.logger = logging.getLogger("bcs.modeling.ModelEvaluator")

        self.df = df
        self._identify_feature_columns()
        self.sdf = df[self.feature_columns + [config.target_column, config.cv_column]]
        self.categorical_columns = ModelEvaluator.convert_to_categorical_columns(self.df, self.feature_columns)

    def _identify_feature_columns(self):
        feature_columns = [
            col
            for col in self.df.columns
            if col.startswith("BM") or col.startswith("status_") or col.startswith("demographics_")
        ]
        self.logger.debug(f"{len(feature_columns)} feature columns before removal")
        # remove manually-excluded columns
        if len(self.config.excluded_columns) > 0:
            feature_columns = [col for col in feature_columns if col not in self.config.excluded_columns]
        if self.config.patient_feature_action == "exclude":
            feature_columns = [col for col in feature_columns if col.startswith("BM")]
        if self.config.biomarker_feature_action == "exclude":
            feature_columns = [col for col in feature_columns if not col.startswith("BM")]
        elif self.config.biomarker_feature_action == "svd":
            pass  # we need to perform SVD within the train/test split, so we need the raw BM features
        elif self.config.biomarker_feature_action == "keep":
            pass  # leave biomarker columns as-is
        else:
            raise ValueError(f"Unknown config value: {self.config.biomarker_feature_action=}")
        self.logger.info(f"{len(feature_columns)} feature columns after removal")
        if len(feature_columns) == 0:
            raise ValueError("Configuration resulted in no features.")
        self.feature_columns = feature_columns

    def train_and_evaluate(self):
        model = GbmModel(self.config)
        model.fit_eval(self.df, self.feature_columns, self.categorical_columns)
        return model

    @staticmethod
    def convert_to_categorical_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
        categorical_columns = list(df[feature_columns].select_dtypes(include="object").columns)
        for categorical_column in categorical_columns:
            if categorical_column == "status_disease_sub_type":  # config.is_disease_sub_type_ordered
                ordered_cat = CategoricalDtype(sorted(df[categorical_column].unique()), ordered=True)
                df[categorical_column] = df[categorical_column].astype(ordered_cat)
            else:
                df[categorical_column] = df[categorical_column].astype("category")
        return categorical_columns


class GbmModel:
    def __init__(self, config: ModelingConfig):
        self.config = config

    def fit_eval(self, df: pd.DataFrame, all_feature_columns: list[str], categorical_columns: list[str]):
        """Fit this GBM model and evaluate it.
        Holds out according to config.cv_column.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with all of the feature_columns
        all_feature_columns : list[str]
            Columns to use in df for modeling.
        categorical_columns : list[str]
            A subset of all_feature_columns indicating categorical variables.
        """
        y_true_all = df.target_label.copy().rename("y_true")
        y_pred_all = df.target_label.copy().rename("y_pred")
        y_score_all = df.target_label.astype("float").rename("y_score")

        biomarker_columns = [col for col in all_feature_columns if col.startswith("BM")]
        nonbiomarker_columns = [col for col in all_feature_columns if col not in biomarker_columns]

        metrics_list = []
        feature_columns = all_feature_columns[:]
        for iname, valid_df in df.groupby(self.config.cv_column):
            train_df = df[~df.index.isin(valid_df.index)]
            assert len(train_df) + len(valid_df) == len(df)
            if self.config.biomarker_feature_action == "svd":
                pca = sklearn.decomposition.PCA(n_components=self.config.biomarker_svd_n_components)
                train_transformed = pca.fit_transform(train_df[biomarker_columns].fillna(0))
                valid_transformed = pca.transform(valid_df[biomarker_columns].fillna(0))
                train_pca_df = pd.DataFrame(train_transformed, index=train_df.index).rename(
                    columns=lambda col: f"BM_pca{col}",
                )
                valid_pca_df = pd.DataFrame(valid_transformed, index=valid_df.index).rename(
                    columns=lambda col: f"BM_pca{col}",
                )
                feature_columns = nonbiomarker_columns + list(train_pca_df.columns)
                train_df = pd.merge(train_df, train_pca_df, how="left", left_index=True, right_index=True)
                valid_df = pd.merge(valid_df, valid_pca_df, how="left", left_index=True, right_index=True)

            column_encoder = GbmModel.get_encoder()
            X_train = column_encoder.fit_transform(train_df[feature_columns])
            X_valid = column_encoder.transform(valid_df[feature_columns])
            X_train = pd.DataFrame(X_train, index=train_df.index, columns=column_encoder.get_feature_names_out())
            X_valid = pd.DataFrame(X_valid, index=valid_df.index, columns=column_encoder.get_feature_names_out())

            clf = sklearn.ensemble.HistGradientBoostingClassifier(
                categorical_features=categorical_columns,
                learning_rate=self.config.gbm_learning_rate,
                max_iter=self.config.gbm_max_iter,
                max_leaf_nodes=self.config.gbm_max_leaf_nodes,
                min_samples_leaf=self.config.gbm_min_samples_leaf,
                l2_regularization=self.config.gbm_l2_regularization,
            )
            clf.fit(X_train, train_df.target_label)

            y_pred = clf.predict(X_valid)
            y_score = clf.predict_proba(X_valid)[:, 1]
            y_true = valid_df.target_label

            y_pred_all[valid_df.index] = y_pred
            y_score_all[valid_df.index] = y_score

            metrics = {
                "experiment_name": self.config.experiment_name,
                "institution_name": iname,
                **GbmModel.get_metrics(y_true, y_pred, y_score),
            }
            metrics_list.append(metrics)
        metrics = {
            "experiment_name": self.config.experiment_name,
            "institution_name": "All",
            **GbmModel.get_metrics(y_true_all, y_pred_all, y_score_all),
        }
        metrics_list.append(metrics)
        # following sklearn conventions, we only define these attributes after the model is fit
        self.metrics_ = metrics_list
        self.preds_ = y_pred_all
        self.scores_ = y_score_all

    @staticmethod
    def get_metrics(y_true, y_pred, y_score):
        metrics = {
            "n": len(y_true),
            "n_pos": y_true.sum(),
            "n_pos_pred": y_pred.sum(),
            "acc": (y_pred == y_true).sum() / len(y_true),
            "f1_pos": sklearn.metrics.f1_score(y_true, y_pred),
            "roc_auc": sklearn.metrics.roc_auc_score(y_true, y_score),
        }
        return metrics

    @staticmethod
    def get_encoder():
        ordinal_encoder = sklearn.compose.make_column_transformer(
            (
                sklearn.preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                sklearn.compose.make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",  # keep non-categorical columns after transformation
            verbose_feature_names_out=False,  # keep original feature names
        )
        return ordinal_encoder
