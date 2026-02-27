from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Wine dataset and return features, labels, and metadata.
    Returns:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
        feature_names (list[str]): Feature names.
        target_names (list[str]): Class names.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names.tolist()
    return X, y, feature_names, target_names

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
