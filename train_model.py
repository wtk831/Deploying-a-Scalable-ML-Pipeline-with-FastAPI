
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census dataset.
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")

print(f"Loading dataset from: {data_path}")
data = pd.read_csv(data_path)

# Split the data into training and testing datasets.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"]
)

print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Preprocess the training dataset and fit the encoders.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Preprocess the testing dataset using the fitted encoders.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the machine learning model.
print("Training Random Forest model...")
model = train_model(X_train, y_train)

# Create the model directory if it does not exist.
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Save the trained model.
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

# Save the fitted categorical encoder.
encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

# Save the fitted label binarizer.
lb_path = os.path.join(model_dir, "lb.pkl")
save_model(lb, lb_path)

print("Model and encoders saved successfully.")

# Load the saved model.
model = load_model(model_path)

# Generate predictions on the testing dataset.
preds = inference(model, X_test)

# Calculate and print overall model performance.
p, r, fb = compute_model_metrics(y_test, preds)

print("\nOverall Model Performance")
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute model performance on categorical slices.
slice_path = os.path.join(project_path, "slice_output.txt")

with open(slice_path, "w", encoding="utf-8") as f:
    # Iterate through the categorical features.
    for col in cat_features:

        # Iterate through the unique values of each feature.
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]

            p, r, fb = performance_on_categorical_slice(
                test,
                col,
                slicevalue,
                cat_features,
                "salary",
                encoder,
                lb,
                model,
            )

            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}",
                file=f,
            )
            print(file=f)

print("\nCategorical slice results saved to slice_output.txt.")
print("Model training and evaluation completed successfully.")
