
# Model Card

For additional information see the Model Card paper:
[https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

Random Forest Classifier developed using Python and Scikit-learn 1.5.1. Designed to predict if an individual earns more than $50,000 per year based on census data.

Parameters:

- Number of decision trees: 100
- Random state: 42
- Number of parallel jobs: -1

Categorical features were processed using OneHotEncoder. Salary labels were converted into binary values using LabelBinarizer. The trained model and fitted encoders were
saved using pickle for future inference and API deployment.

## Intended Use

The model is for educational purposes and demonstrates the development and deployment of a machine learning pipeline using FastAPI.

The model predicts one of two salary categories:

- <=50K: Annual income of $50,000 or less.
- >50K: Annual income greater than $50,000.

The model is not to be used for employment, lending,
insurance, or other high-impact financial decisions regarding individuals.

## Training Data

The model was trained using the census dataset provided for this project. The original dataset contains 32,561 records and 15 columns, including the salary target variable.

An 80/20 stratified train-test split was performed using a random state of 42. The training dataset contains 26,048 records.

The dataset includes demographic, educational, employment,and financial attributes.

The following categorical features were one-hot encoded:

- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

Missing information represented by a question mark (?) was retained as an unknown categorical value. Numerical features were not scaled because Random Forest does not require feature scaling.

## Evaluation Data

The model was evaluated with the remaining 20% of the
census dataset, consisting of 6,513 records.

The evaluation data was not used during model training. The fitted encoders from the training dataset were reused for processing the evaluation data and prevent data leakage.

Model performance was evaluated on the complete test dataset and on individual slices of the categorical features.

Categorical slice results are available in
slice_output.txt.

## Metrics

The model was evaluated using precision, recall, and F1 score. The positive class represents individuals earning more than
$50,000 per year.

| Metric | Score |
|--------|-------|
| Precision | 0.7353 |
| Recall | 0.6378 |
| F1 Score | 0.6831 |

Precision measures the proportion of positive predictions that were correct.

Recall measures the proportion of actual positive cases correctly identified.

The F1 score provides a balance between precision and recall.

Selected categorical slice results:

| Workclass | Count | Precision | Recall | F1 Score |
|-----------|------:|----------:|-------:|---------:|
| Private | 4,597 | 0.7354 | 0.6254 | 0.6760 |
| Federal-gov | 195 | 0.7324 | 0.6842 | 0.7075 |
| Self-emp-inc | 222 | 0.7541 | 0.7797 | 0.7667 |
| Unknown (?) | 363 | 0.5517 | 0.4444 | 0.4923 |
| Never-worked | 1 | 1.0000 | 1.0000 | 1.0000 |

These results demonstrate that model performance varies over different employment categories.

## Ethical Considerations

The census dataset contains sensitive demographic information of age, race, sex, and native country.

Historical census data could possibly reflect existing social and economic inequalities. Therefore, the model may reproduce biases present in the training data.

Performance differences across demographic and employment groups should be considered when interpreting predictions.
Categorical slice evaluation helps identify these differences but does not establish that the model is fair or unbiased.

The model should not be used to discriminate against
individuals or make decisions that affect access to employment, financial services, or other opportunities.

## Caveats and Recommendations

The model has several limitations:

- The census dataset may not reflect current economic
  conditions or income distributions.
- Missing information represented by question marks  may   negatively affect model performance.
- Some categorical groups contain very few records,
  making their performance metrics unreliable.
- The Never-worked category contains only one       evaluation record, so its perfect scores should be interpreted cautiously.
- The metric calculation uses zero_division=1, which may produce misleadingly high scores for some categories when metrics are undefined.
- The model was evaluated using a single train-test split rather than cross-validation.

Future improvements could include hyperparameter tuning, cross-validation, additional fairness testing, improved handling of missing values, and evaluation using more recent and representative census data.

The model should remain limited to educational and
demonstration purposes without performing further validation.
