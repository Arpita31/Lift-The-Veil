# Unredactor

Name: Arpita Patnaik
cis6930fa24 -- Project 2

# Project Description
This project implements a machine learning-based unredactor system for movie reviews. The system uses a Random Forest classifier to predict redacted names in movie reviews based on contextual features. Find the detailed project description on [https://ufdatastudio.com/cis6930fa24/assignments/project2].

# How to Install
```bash
# Create and activate virtual environment
pipenv install -e .

# Install required packages
pipenv install pandas numpy scikit-learn nltk

# Download required NLTK data
pipenv run python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words
```

# How to Run
To run the unredactor:
```bash
pipenv run python unredactor.py
```

To run the tests:
```bash
pipenv run pytest test/test_random.py -v
```

# Sample Output
```
=== Movie Review Unredactor ===

Loaded 200 training examples.

Training Random Forest...

Validation Accuracy for Random Forest: 17.50%

Classification Report:
               precision    recall  f1-score   support

Alice Johnson       0.00      0.00      0.00         4
    Bob Brown       0.20      0.33      0.25         3
Charlie White       0.00      0.00      0.00         3
  Diana Black       0.00      0.00      0.00         4
 Edward Green       1.00      0.20      0.33        10
   Fiona Grey       0.00      0.00      0.00         3
   Jane Smith       0.25      0.33      0.29         6
     John Doe       0.13      0.29      0.18         7

     accuracy                           0.17        40
    macro avg       0.20      0.14      0.13        40
 weighted avg       0.33      0.17      0.18        40

Loaded 200 test examples.

Generating submission file...
Submission file saved as submission.tsv
```

# Functions

1. `BaseUnredactor.__init__(classifier_name, classifier)`: Initializes base unredactor model with specified classifier.
2. `BaseUnredactor.get_context_features(context)`: Extracts features from text context including:
   - Context length
   - Word count
   - Punctuation presence
   - Capitalization patterns
   - Redaction symbol count
   - Average word length
3. `BaseUnredactor.extract_features(df)`: Generates feature representations for each row in dataset.
4. `BaseUnredactor.fit(train_df)`: Trains model using provided training data.
5. `BaseUnredactor.predict(context)`: Makes prediction for given text context.
6. `BaseUnredactor.evaluate(val_df)`: Assesses model performance using validation data.
7. `RandomForestUnredactor`: Implementation of base unredactor using Random Forest classifier.
8. `load_training_data(file_path)`: Loads and preprocesses training data from TSV file.
9. `load_test_data(file_path)`: Loads and preprocesses test data from TSV file.
10. `generate_submission(model, test_data, output_file)`: Creates submission file with predictions.
11. `main()`: Orchestrates the entire process including:
    - Data loading
    - Model training
    - Evaluation
    - Submission generation

# Bugs and Assumptions

### Assumptions:
1. Training data is provided in TSV format with columns: split, name, context
2. Test data is provided in TSV format with columns: id, context
3. Redacted names in context are marked with █ symbols
4. Context provides sufficient information for name prediction
5. Names in training data are representative of test data

### Known Limitations:
1. Model relies heavily on context features and may struggle with:
   - Unusual or rare names
   - Limited context information
   - Multiple redacted names in same context
2. Feature extraction assumes well-formed input:
   - Valid text strings
   - Proper redaction marking
   - Consistent formatting
3. Error handling for edge cases:
   - Empty strings
   - None values
   - Non-string inputs
   - Malformed data

### Error Handling:
1. Input validation for:
   - None values
   - Empty strings
   - Non-string inputs
   - Invalid file paths
2. Default feature values for invalid inputs
3. Exception handling during:
   - Data loading
   - Feature extraction
   - Model training
   - Prediction generation

[Previous sections remain the same until Functions]

# Pipeline Instructions

The unredactor system follows this pipeline:

1. **Data Loading and Preprocessing**:
   - Training data is loaded from 'unredactor.tsv' with three columns:
     - split (train/test designation)
     - name (actual name that was redacted)
     - context (text with redacted names marked by █)
   - Test data is loaded from 'test.tsv' with two columns:
     - id (unique identifier)
     - context (text with redacted names)

2. **Feature Extraction**:
   The system extracts the following features from each context:
   - context_length: Total length of the text
   - context_word_count: Number of words
   - has_question_mark: Presence of question marks
   - has_exclamation: Presence of exclamation marks
   - starts_with_capital: Whether first word is capitalized
   - capital_word_ratio: Proportion of capitalized words
   - redacted_symbol_count: Number of █ symbols
   - average_word_length: Mean length of words

3. **Model Training**:
   - Data is split into training (80%) and validation (20%) sets
   - Features are vectorized using DictVectorizer
   - Names are encoded using LabelEncoder
   - Random Forest classifier is trained with parameters:
     - n_estimators=100
     - max_depth=10
     - random_state=42

4. **Model Evaluation**:
   - Model is evaluated on validation set
   - Metrics reported include:
     - Overall accuracy
     - Per-class precision, recall, f1-score
     - Support for each class

5. **Prediction Generation**:
   - Model makes predictions on test data
   - Predictions are saved in TSV format with columns:
     - id (from test data)
     - name (predicted name)

# Pipeline Workflow Example:
```python
# 1. Load Data
train_df = load_training_data('unredactor.tsv')
train_data, val_data = train_test_split(train_df, test_size=0.2)

# 2. Initialize and Train Model
model = RandomForestUnredactor()
model.fit(train_data)

# 3. Evaluate Performance
accuracy = model.evaluate(val_data)

# 4. Generate Predictions
test_df = load_test_data('test.tsv')
generate_submission(model, test_df, 'submission.tsv')
```

