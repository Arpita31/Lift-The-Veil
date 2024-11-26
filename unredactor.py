import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading required NLTK data...")
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

class BaseUnredactor:
    """Base class for all unredactor models."""
    
    def __init__(self, classifier_name, classifier):
        """
        Initialize the base unredactor model with a specific classifier.

        Args:
            classifier_name (str): A string representing the name of the classifier.
            classifier (sklearn classifier): The machine learning classifier to be used.

        Returns:
            None
        """
        self.classifier_name = classifier_name
        self.vectorizer = DictVectorizer(sparse=False)
        self.classifier = classifier
        self.label_encoder = LabelEncoder()
        
    # Modified get_context_features method in BaseUnredactor class
    def get_context_features(self, context):
        """
        Extract key characteristics from the provided text context.

        Args:
            context (str): The input text containing the redacted information.

        Returns:
            dict: A dictionary containing various extracted features from the context.
        """
        try:
            context = str(context) if context is not None else ""
            words = context.split()
            features = {
                'context_length': len(context),
                'context_word_count': len(words),
                'has_question_mark': '?' in context,
                'has_exclamation': '!' in context,
                'starts_with_capital': bool(words and words[0][0].isupper()) if words else False,
                'capital_word_ratio': sum(1 for w in words if w and w[0].isupper()) / len(words) if words else 0,
                'redacted_symbol_count': context.count('█'),
                'average_word_length': np.mean([len(w) for w in words]) if words else 0
            }
        except Exception as e:
            # Return default features for invalid input
            features = {
                'context_length': 0,
                'context_word_count': 0,
                'has_question_mark': False,
                'has_exclamation': False,
                'starts_with_capital': False,
                'capital_word_ratio': 0,
                'redacted_symbol_count': 0,
                'average_word_length': 0
            }
        return features
    
    def extract_features(self, df):
        """
        Generate feature representations for each row in the dataset.

        Args:
            df (pd.DataFrame): A DataFrame containing the text data for feature extraction.

        Returns:
            list: A list of dictionaries, each representing features for a specific row.
        """
        return [self.get_context_features(row['context']) for _, row in df.iterrows()]

    def fit(self, train_df):
        """
        Train the model using the provided training dataset.

        Args:
            train_df (pd.DataFrame): The training data containing text contexts and labels.

        Returns:
            None
        """
        print(f"\nTraining {self.classifier_name}...")
        X = self.vectorizer.fit_transform(self.extract_features(train_df))
        y = train_df['name']
        y_encoded = self.label_encoder.fit_transform(y)
        self.classifier.fit(X, y_encoded)

    def predict(self, context):
        """
        Make a prediction for the given text context.

        Args:
            context (str): The input text for which a prediction is required.

        Returns:
            str: The predicted label or name corresponding to the context.
        """
        features = self.vectorizer.transform([self.get_context_features(context)])
        prediction = self.classifier.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]

    def evaluate(self, val_df):
        """
        Assess the model's performance using a validation dataset.

        Args:
            val_df (pd.DataFrame): The validation data to evaluate the model's accuracy.

        Returns:
            float: The accuracy of the model on the validation data.
        """
        X_val = self.vectorizer.transform(self.extract_features(val_df))
        y_val = val_df['name']
        y_encoded = self.label_encoder.transform(y_val)
        y_pred = self.classifier.predict(X_val)
        accuracy = accuracy_score(y_encoded, y_pred)
        print(f"\nValidation Accuracy for {self.classifier_name}: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_encoded, y_pred, target_names=self.label_encoder.classes_))
        return accuracy

class RandomForestUnredactor(BaseUnredactor):
    def __init__(self):
        """
        Initialize a Random Forest unredactor model.

        Args:
            None

        Returns:
            None
        """
        super().__init__(
            "Random Forest",
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        )

def load_training_data(file_path='unredactor.tsv'):
    """
    Load and preprocess training data from a TSV file.

    Args:
        file_path (str): Path to the training data file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed training data.
    """
    df = pd.read_csv(file_path, sep='\t', names=['split', 'name', 'context'])
    print(f"Loaded {len(df)} training examples.")
    return df

def load_test_data(file_path='test.tsv'):
    """
    Load and preprocess test data from a TSV file.

    Args:
        file_path (str): Path to the test data file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed test data.
    """
    df = pd.read_csv(file_path, sep='\t', names=['id', 'context'])
    print(f"Loaded {len(df)} test examples.")
    return df

def generate_submission(model, test_data, output_file='submission.tsv'):
    """
    Create a submission file by predicting labels for the test data.

    Args:
        model (BaseUnredactor): The trained model used for generating predictions.
        test_data (pd.DataFrame): The test dataset containing text contexts.
        output_file (str): The file name or path to save the submission file.

    Returns:
        None
    """
    print("\nGenerating submission file...")
    predictions = []
    for _, row in test_data.iterrows():
        pred = model.predict(row['context'])
        predictions.append({'id': row['id'], 'name': pred})
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_file, sep='\t', index=False)
    print(f"Submission file saved as {output_file}")

def main():
    """
    The main function orchestrating the data loading, model training, evaluation, and submission generation.

    Args:
        None

    Returns:
        None
    """
    print("=== Movie Review Unredactor ===\n")
    
    # Load training data
    train_df = load_training_data('unredactor.tsv')
    
    # Split into training and validation sets
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestUnredactor()
    rf_model.fit(train_data)
    
    # Evaluate the model
    rf_model.evaluate(val_data)
    
    # Load test data
    test_df = load_test_data('test.tsv')
    
    # Generate submission file
    generate_submission(rf_model, test_df, 'submission.tsv')

if __name__ == "__main__":
    main()
