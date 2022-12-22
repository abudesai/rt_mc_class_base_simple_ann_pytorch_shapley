import numpy as np, pandas as pd
import os, sys
import json
import pprint
from shap import Explainer

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.mc_classifier as mc_classifier


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.data_schema = data_schema
        self.preprocessor = None
        self.model = None
        self.id_field_name = self.data_schema["inputDatasets"][
            "multiClassClassificationBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = mc_classifier.load_model(self.model_path)
        return self.model

    def _get_predictions(self, data, return_probs=True):
        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X, pred_ids, features = (
            proc_data["X"].astype(np.float),
            proc_data["ids"],
            proc_data["features"],
        )
        # make predictions
        if return_probs:
            preds = model.predict(pred_X)
        else:
            preds = model.predict(pred_X)

        return preds, pred_ids, features

    def predict_proba(self, data):
        preds, pred_ids, features = self._get_predictions(data, return_probs=True)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        id_df = pd.DataFrame(pred_ids, columns=[self.id_field_name])

        # return the prediction df with the id and class probability fields
        preds_df = pd.concat([id_df, pd.DataFrame(preds, columns=class_names)], axis=1)
        return preds_df

    def predict(self, data):
        preds_df = self.predict_proba(data)
        class_names = [str(c) for c in preds_df.columns[1:]]
        preds_df["prediction"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)
        preds_df.drop(class_names, axis=1, inplace=True)
        return preds_df

    def _get_target_class_proba(self, X):
        """
        Returns predicted probability of the target class
        """
        model = self._get_model()
        preds = model.predict(X)
        return preds

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data.head(self.MAX_LOCAL_EXPLANATIONS))
        # ------------------------------------------------------------------------------
        # original class predictions
        model = self._get_model()
        pred_X, ids, features = (
            proc_data["X"].astype(np.float),
            proc_data["ids"],
            proc_data["features"],
        )
        pred_class_probs = model.predict(pred_X)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)

        # ------------------------------------------------------------------------------
        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        # create the shapley explainer
        mask = np.zeros_like(pred_X)
        explainer = Explainer(self._get_target_class_proba, mask, seed=1)
        # Get local explanations
        shap_values = explainer(pred_X)
        # ------------------------------------------------------------------------------
        # create json objects of explanation scores
        N = pred_X.shape[0]
        explanations = []
        for i in range(N):
            sample_expl_dict = {}
            sample_expl_dict[self.id_field_name] = ids[i]
            sample_expl_dict["predicted_class_prob"] = pred_class_probs[i].max()
            sample_expl_dict["predicted_class"] = class_names[
                pred_class_probs[i].argmax()
            ]

            # pprint.pprint(sample_expl_dict) ; sys.exit()
            sample_expl_dict["class_explanations"] = {}
            for j, c in enumerate(class_names):
                class_exp_dict = {}
                class_exp_dict["predicted_class_prob"] = pred_class_probs[i, j]
                class_exp_dict["base_probability"] = shap_values.base_values[i, j]
                feature_impacts = {
                    f: np.round(v, 4)
                    for f, v in zip(features, shap_values.values[i, :, j])
                }
                class_exp_dict["feature_impacts"] = feature_impacts

                sample_expl_dict["class_explanations"][str(c)] = class_exp_dict
            explanations.append(sample_expl_dict)

        # ------------------------------------------------------
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
