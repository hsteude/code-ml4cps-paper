import joblib
from kserve import Model, constants
from kserve.errors import InferenceError
from typing import Dict
import logging
import os
from sklearn.base import TransformerMixin
from torch.nn.modules.module import Module
from torch import Tensor, jit
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
        processed = pd.DataFrame(self.scaler.transform(data), columns=data.columns).T.to_dict()
        # KServe validator expects a list for value of instances.
        # Any dicts have to be json serializable, so column names have to be str
        return {'instances': [processed[x] for x in processed]}

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
        # torch script models
        self.model = jit.load(f'/mnt/models/{os.environ["MODEL_FILENAME"]}')
        self.ready = True
        print(f"Loaded {self.model.__str__()}")
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        data = pd.DataFrame(payload['instances'])
        try:
            return {"predictions": self.model(Tensor(data.values)).detach().numpy().tolist()}
        except Exception as e:
            raise InferenceError(str(e))
