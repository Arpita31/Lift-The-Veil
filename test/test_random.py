import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from unredactor import (
    BaseUnredactor,
    RandomForestUnredactor,
    load_training_data,
    load_test_data,
    generate_submission
)

# ============= Fixtures =============
@pytest.fixture
def sample_training_data():
    """
    Fixture providing sample training data.

    Args:
        None

    Returns:
        pd.DataFrame: Sample training data with columns ['split', 'name', 'context']
    """
    return pd.DataFrame({
        'split': ['train', 'train', 'test'],
        'name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
        'context': [
            'A movie starring █████████ was released.',
            'Director █████████ won an award.',
            'The actor █████████ gave a great performance.'
        ]
    })

@pytest.fixture
def sample_test_data():
    """
    Fixture providing sample test data.

    Args:
        None

    Returns:
        pd.DataFrame: Sample test data with columns ['id', 'context']
    """
    return pd.DataFrame({
        'id': [1, 2],
        'context': [
            'A movie starring █████████ was released.',
            'Director █████████ won an award.'
        ]
    })

@pytest.fixture
def rf_unredactor(sample_training_data):
    """
    Fixture providing a trained RandomForestUnredactor instance.

    Args:
        sample_training_data (pd.DataFrame): Sample training data for model fitting

    Returns:
        RandomForestUnredactor: Trained instance of RandomForestUnredactor
    """
    model = RandomForestUnredactor()
    model.fit(sample_training_data)
    return model

# ============= Test Cases =============
class TestRandomForestUnredactor:
    """Test cases for RandomForestUnredactor class."""
    
    def test_initialization(self):
        """
        Test initialization of RandomForestUnredactor.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If initialization parameters don't match expected values
        """
        model = RandomForestUnredactor()
        assert model.classifier_name == "Random Forest"
        assert isinstance(model.classifier, RandomForestClassifier)
    
    def test_feature_extraction(self, sample_training_data):
        """
        Test feature extraction from context.

        Args:
            sample_training_data (pd.DataFrame): Sample training data

        Returns:
            None

        Raises:
            AssertionError: If extracted features don't match expected format
        """
        model = RandomForestUnredactor()
        context = sample_training_data['context'].iloc[0]
        features = model.get_context_features(context)
        
        assert isinstance(features, dict)
        assert 'context_length' in features
        assert 'redacted_symbol_count' in features
        assert features['redacted_symbol_count'] == 9
    
    def test_fit_predict(self, sample_training_data):
        """
        Test model fitting and prediction.

        Args:
            sample_training_data (pd.DataFrame): Sample training data

        Returns:
            None

        Raises:
            AssertionError: If prediction doesn't match expected format or values
        """
        model = RandomForestUnredactor()
        model.fit(sample_training_data)
        prediction = model.predict(sample_training_data['context'].iloc[0])
        assert isinstance(prediction, str)
        assert prediction in sample_training_data['name'].values

def test_load_training_data(tmp_path):
    """
    Test loading training data from file.

    Args:
        tmp_path (Path): pytest fixture providing temporary directory path

    Returns:
        None

    Raises:
        AssertionError: If loaded data doesn't match expected format
    """
    # Create temporary training file
    file_path = tmp_path / "temp_train.tsv"
    pd.DataFrame({
        'split': ['train'],
        'name': ['John Doe'],
        'context': ['Test context']
    }).to_csv(file_path, sep='\t', index=False, header=False)
    
    df = load_training_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['split', 'name', 'context']
    assert len(df) > 0

def test_load_test_data(tmp_path):
    """
    Test loading test data from file.

    Args:
        tmp_path (Path): pytest fixture providing temporary directory path

    Returns:
        None

    Raises:
        AssertionError: If loaded data doesn't match expected format
    """
    # Create temporary test file
    file_path = tmp_path / "temp_test.tsv"
    pd.DataFrame({
        'id': [1],
        'context': ['Test context']
    }).to_csv(file_path, sep='\t', index=False, header=False)
    
    df = load_test_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['id', 'context']
    assert len(df) > 0

def test_generate_submission(rf_unredactor, sample_test_data, tmp_path):
    """
    Test generating submission file.

    Args:
        rf_unredactor (RandomForestUnredactor): Trained unredactor model
        sample_test_data (pd.DataFrame): Sample test data
        tmp_path (Path): pytest fixture providing temporary directory path

    Returns:
        None

    Raises:
        AssertionError: If generated file doesn't match expected format
    """
    output_file = tmp_path / "submission.tsv"
    generate_submission(rf_unredactor, sample_test_data, str(output_file))
    
    assert output_file.exists()
    submission_df = pd.read_csv(output_file, sep='\t')
    assert list(submission_df.columns) == ['id', 'name']
    assert len(submission_df) == len(sample_test_data)

def test_model_validation(rf_unredactor, sample_training_data):
    """
    Test model validation metrics.

    Args:
        rf_unredactor (RandomForestUnredactor): Trained unredactor model
        sample_training_data (pd.DataFrame): Sample training data

    Returns:
        None

    Raises:
        AssertionError: If validation metrics don't meet expectations
    """
    # Test on training data for basic validation
    predictions = []
    for _, row in sample_training_data.iterrows():
        pred = rf_unredactor.predict(row['context'])
        predictions.append(pred)
    
    # Check if predictions are valid
    assert len(predictions) == len(sample_training_data)
    assert all(isinstance(pred, str) for pred in predictions)
    assert all(pred in sample_training_data['name'].values for pred in predictions)

def test_error_handling():
    """
    Test error handling in model.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If error handling doesn't work as expected
    """
    model = RandomForestUnredactor()
    
    # Test with invalid context
    empty_context = ""
    features = model.get_context_features(empty_context)
    assert features['context_length'] == 0
    assert features['redacted_symbol_count'] == 0
    
    # Test with None context
    none_features = model.get_context_features(None)
    assert isinstance(none_features, dict)
    assert none_features['context_length'] == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])