import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.ensemble
import sklearn.metrics
import sklearn.preprocessing
from pandas.api.types import CategoricalDtype


@dataclass
class ModelingConfig:
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
            raise NotImplementedError("No reduction yet.")
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
        return model.metrics_list

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
        self.metrics_list = []

    def fit_eval(self, df: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str]):
        for iname, valid_df in df.groupby(self.config.cv_column):
            column_encoder = GbmModel.get_encoder()
            train_df = df[~df.index.isin(valid_df.index)]
            assert len(train_df) + len(valid_df) == len(df)
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

            metrics = {
                "experiment_name": self.config.experiment_name,
                "institution_name": iname,
                **GbmModel.get_metrics(y_true, y_pred, y_score),
            }
            self.metrics_list.append(metrics)
        return self.metrics_list

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
