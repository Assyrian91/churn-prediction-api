import mlflow
from mlflow.tracking import MlflowClient
import pickle
import json
from datetime import datetime
import os

class ModelRegistry:
    def __init__(self, model_name="churn-prediction-model"):
        self.model_name = model_name
        self.client = MlflowClient()
        
    def register_model(self, run_id=None):
        if run_id is None:
            experiments = self.client.search_experiments()
            if experiments:
                runs = self.client.search_runs(experiment_ids=[experiments[0].experiment_id])
                if runs:
                    run_id = runs[0].info.run_id
                else:
                    print("No runs found!")
                    return None
        
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_version = mlflow.register_model(model_uri, self.model_name)
            print(f"Model registered: {self.model_name} version {model_version.version}")
            return model_version
        except Exception as e:
            print(f"Error registering model: {e}")
            return None
    
    def promote_to_production(self, version):
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production"
        )
        print(f"Model version {version} promoted to Production")
    
    def get_latest_production_model(self):
        try:
            latest_version = self.client.get_latest_versions(
                self.model_name, 
                stages=["Production"]
            )
            if latest_version:
                return latest_version[0]
            return None
        except:
            return None

if __name__ == "__main__":
    registry = ModelRegistry()
    model_version = registry.register_model()
    if model_version:
        registry.promote_to_production(model_version.version)
