import joblib
from kserve import Model, constants
from kserve.errors import InferenceError, ModelMissingError
from typing import Dict
import logging
import os
from container_component_src.model.lightning_module import TimeStampVAE
from sklearn.base import TransformerMixin
from torch.nn.modules.module import Module
from torch import Tensor
import pandas as pd

logging.basicConfig(level=constants.KSERVE_LOGLEVEL)


class CustomTransformer(Model):
    scaler: TransformerMixin

    def __init__(self, name: str, predictor_host: str, headers: Dict[str, str] = None):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = False
        self.load()

    def load(self):
        self.scaler = joblib.load(f'/mnt/models/{os.environ["SCALER_FILENAME"]}')
        self.ready = True
        print(f"Loaded {self.scaler.__str__()}")
        return self.ready

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        data = pd.DataFrame(inputs['instances'])
        return {'instances': pd.DataFrame(self.scaler.transform(data), columns=data.columns).to_dict()}

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        return inputs


class CustomPredictor(Model):
    model: Module

    def __init__(self, name: str, device: str = 'cpu'):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.device = device
        self.load()

    def load(self):
        self.model = TimeStampVAE.load_from_checkpoint(f'/mnt/models/{os.environ["MODEL_FILENAME"]}').to(self.device)
        self.ready = True
        print(f"Loaded {self.model.__str__()}")
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        data = pd.DataFrame(payload['instances'])

        try:
            mean_z, log_var_z, mean_x, log_var_x, neg_log_likelihood = \
                self.model.infer(Tensor(data.values))
            return {"mean_z": mean_z.detach().numpy().tolist(),
                    "log_var_z": log_var_z.detach().numpy().tolist(),
                    "mean_x": mean_x.detach().numpy().tolist(),
                    "log_var_x": log_var_x.detach().numpy().tolist(),
                    "neg_log_likelihood": neg_log_likelihood.detach().numpy().tolist()}
        except Exception as e:
            raise InferenceError(str(e))
