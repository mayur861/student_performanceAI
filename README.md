# StudentIQ — Student Performance Prediction

An end-to-end Machine Learning project that predicts a student's **Mathematics Score** using demographic, family/education, lunch, test-preparation, reading and writing information.

## Architecture

```text
                 ┌──────────────────────┐
                 │      stud.csv        │
                 │    1,000 records     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Validation      │
                 │ types / null checks  │
                 └──────────┬───────────┘
                            │
                            ▼
          ┌─────────────────┴─────────────────┐
          │        Feature Separation         │
          │ Numeric            Categorical    │
          └──────────┬─────────────┬──────────┘
                     │             │
             ┌───────▼──────┐ ┌────▼────────────┐
             │ Imputer      │ │ Imputer         │
             │ Median       │ │ Most Frequent  │
             │ + Scaler     │ │ + OneHotEncoder │
             └───────┬──────┘ └────┬────────────┘
                     └───────┬─────┘
                             ▼
                    ColumnTransformer
                             │
                             ▼
                    Regression Model
                             │
                             ▼
                  Evaluation / Selection
                             │
                             ▼
             student_performance_pipeline.joblib
                             │
                             ▼
                       Flask Web App
                             │
                             ▼
                    Attractive UI Result
```

## Dataset

Input columns:

- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score

Target:

- `math_score`

## ML Pipeline

The project uses a single Scikit-learn `Pipeline` so preprocessing and the model are saved together.

### Numeric pipeline
`SimpleImputer(strategy="median") → StandardScaler()`

### Categorical pipeline
`SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")`

### Combined
`ColumnTransformer → Regression Model`

Models compared:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Extra Trees Regressor

The model with the lowest RMSE on the held-out test set is saved automatically.

## Run the project

### 1. Create environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Train

```bash
python train.py
```

This creates:

```text
artifacts/
├── student_performance_pipeline.joblib
├── model_results.csv
└── metadata.json
```

### 4. Start Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Important interview explanation

**Why Pipeline?**

Pipeline prevents preprocessing mismatch between training and prediction. The same fitted imputer, scaler and encoder used during training are automatically applied to new user input.

**Why ColumnTransformer?**

Because the dataset contains both numeric and categorical features, and each type needs different preprocessing.

**Why handle_unknown="ignore"?**

If the Flask UI sends a category not seen during training, the encoder will not crash.

**Why save the whole pipeline?**

Instead of separately saving preprocessing objects and the model, one `.joblib` file contains the complete prediction workflow.

## Project structure

```text
student_performance_prediction/
│
├── data/
│   └── stud.csv
│
├── artifacts/
│   ├── student_performance_pipeline.joblib
│   ├── model_results.csv
│   └── metadata.json
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## Future improvements

- Add SHAP explainability
- Add prediction history
- Add student risk category
- Add SQLite database
- Add Docker deployment
- Deploy on Render/Railway/Azure
- Add REST API endpoint
- Add model monitoring
