
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize the Random Forest classifier.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train the model using the training data.
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """
    Runs model inference and returns the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def save_model(model, path):
    """
    Serializes a model or encoder to a file.

    Inputs
    ------
    model
        Trained machine learning model or encoder.
    path : str
        Path to save pickle file.
    """
    # Save the object using pickle.
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Loads a pickle file from the specified path and returns it.
    """
    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features,
    label, encoder, lb, model
):
    """
    Computes model metrics on a categorical slice of the data.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the sliced feature.
    categorical_features : list
        List containing the names of categorical features.
    label : str
        Name of the target column.
    encoder : OneHotEncoder
        Encoder fitted using the training data.
    lb : LabelBinarizer
        Label binarizer fitted using the training data.
    model : RandomForestClassifier
        Trained machine learning model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Filter the dataset for the specified categorical value.
    data_slice = data[data[column_name] == slice_value]

    # Process the slice using the existing fitted encoders.
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Generate predictions for the categorical slice.
    preds = inference(model, X_slice)

    # Calculate performance metrics for the slice.
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta
