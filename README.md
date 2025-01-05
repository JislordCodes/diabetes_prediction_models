# Diabetic Prediction Models

This repository contains a collection of machine learning models built to predict the likelihood of a patient having diabetes, based on the PIMA Indians Diabetes Database. The models implemented here include:

*   Logistic Regression
*   Random Forest
*   Gradient Boosting (XGBoost)
*   Neural Network

## Dataset

The dataset used in this project is the PIMA Indians Diabetes Database, which can be found in the `diabetes.csv` file. This dataset includes several medical predictor variables and one target variable, 'Outcome'. The predictor variables are:

*   `Pregnancies`: Number of times pregnant
*   `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
*   `BloodPressure`: Diastolic blood pressure (mm Hg)
*   `SkinThickness`: Triceps skin fold thickness (mm)
*   `Insulin`: 2-Hour serum insulin (mu U/ml)
*   `BMI`: Body mass index (weight in kg/(height in m)^2)
*   `DiabetesPedigreeFunction`: Diabetes pedigree function
*   `Age`: Age (years)

The 'Outcome' variable is binary, where 1 indicates diabetes and 0 indicates no diabetes.

## Project Structure

The repository is structured as follows:

*   `diabetes.csv`: The dataset used for training and testing the models.
*   `Gradient_Boosting(XGBoost)_diabetes_prediction.ipynb`: Jupyter Notebook containing the code for the XGBoost model.
*   `Logistic_Regression_diabetes_prediction.ipynb`: Jupyter Notebook containing the code for the Logistic Regression model.
*   `Random_Forest_Model_Diabetes_Prediction.ipynb`: Jupyter Notebook containing the code for the Random Forest model.
*   `neural_net_model_diabtetes_prediction.ipynb`: Jupyter Notebook containing the code for the Neural Network model.
*   `README.md`: This file.

## Model Implementation

Each model is implemented in its own Jupyter Notebook, and follows these basic steps:

1.  **Data Loading and Exploration:** The dataset is loaded, missing values are checked and handled (e.g. filled with the mean), and features and the target are separated.
2.  **Data Splitting:** The data is split into training, validation, and test sets using stratified sampling to ensure the class distribution is preserved.
3.  **Feature Scaling:** The numerical features are standardized using `StandardScaler` to improve model performance.
4.  **Class Imbalance Handling:** The SMOTE (Synthetic Minority Over-sampling Technique) is used to address the class imbalance in the training data.
5.  **Model Training:** The model is trained on the resampled and scaled training data.
6.  **Hyperparameter Tuning (if applicable):**  Models are fine-tuned using cross-validated grid search to find the best hyperparameters
7.  **Model Evaluation:** The models are evaluated using accuracy, confusion matrix, and classification report on the validation and/or testing data.
8.  **Feature Importance:**  Feature importances are calculated and visualized for some models to understand which features are most influential.
9.  **Confusion Matrix Visualization:** Confusion matrices are displayed as heatmaps to analyze model performance in predicting the two classes.
10. **Correlation Heatmap:**  A correlation heatmap is provided to visualize the relationships between different features.

### Details for each model
Here's what is specifically covered for each model:

1.  **Logistic Regression:**
    *   Applies a basic logistic regression model.
    *   Evaluated using accuracy, confusion matrix, and classification report.
    *   Feature importance shown using coeffecient.
    *   Correlation heatmap also provided.

2.  **Random Forest:**
    *   Uses GridSearchCV to tune the model's hyperparameters such as:
        * `n_estimators`: Number of trees
        * `max_depth`: Tree depth
        * `min_samples_split`: Min samples to split
        * `min_samples_leaf`: Min samples per leaf
    *   Evaluated using accuracy, confusion matrix, and classification report.
    *   Feature importance based on importance value.
    *   Correlation heatmap also provided.

3. **Gradient Boosting (XGBoost):**
    *    Basic hyper parameters such as:
        * 'n_estimators': Number of trees
        *  'learning_rate': Step size shrinkage 
        *   'max_depth':  maximum depth of a tree
        *   'min_child_weight': Minimum sum of instance weight needed in a child
        *   'gamma': Minimum loss reduction required to make a further partition on a leaf node
        *   'subsample': Subsample ratio of the training instance
        *  'colsample_bytree': Subsample ratio of columns when constructing each tree
    *  Evaluated using accuracy, confusion matrix, and classification report.
    *   Feature importance based on importance value.

4. **Neural Network:**
    *  A sequential network is built with layers such as:
       *  Dense layers with 'relu' activation function
       * Dropout layers
       * Binary cross-entropy for the loss function
       * Adam optimizer
    *   Validation accuracy, f1-score, confusion matrix and classification report
        are printed for validation set.
    *    Test accuracy, f1-score, confusion matrix and classification report
        are printed for test set.
    *  Feature Importance is analysed using permutation importance

## How to Run the Code

1.  **Clone the repository:** `git clone <repository_url>`
2.  **Install the required libraries:** Run `pip install -r requirements.txt` in your terminal. 
3.  **Open the Jupyter Notebooks:** Use Jupyter Notebook or JupyterLab to open each notebook individually.
4.  **Run the cells:** Execute the cells sequentially in each notebook.
5.  Make sure that you have downloaded the database and placed it in the same location of all ipynb files.

## Model Performance

Results for each model is provided in the output of the notebook.  Test accuracies achieved for each model are:

* Logistic Regression: **75.86%**
* Random Forest:  **78.45%**
* Gradient Boosting(XGBoost): **81.50%**
* Neural Network: **80.17%**

The neural network and XGBoost have the best performance on this dataset.

## Future Work

*   Further hyperparameter tuning for all models.
*   Explore different feature engineering techniques.
*   Implement and compare other machine learning models.
*   Look for different model evaluation metrics.

## Credits

*   **Dataset:** PIMA Indians Diabetes Database, which can be found in the `diabetes.csv` file.
*   **Libraries:** This project utilizes `numpy`, `pandas`, `scikit-learn`, `imblearn`, `tensorflow`, `matplotlib`, `seaborn`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
