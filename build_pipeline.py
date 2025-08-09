import joblib
from sklearn.pipeline import Pipeline

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/churn_prediction_model.pkl")

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

joblib.dump(pipeline, "models/pipeline.pkl")
print("Pipeline saved successfully.")
