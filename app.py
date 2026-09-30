from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "artifacts" / "student_performance_pipeline.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

def performance_label(score):
    if score >= 90:
        return "Outstanding"
    if score >= 80:
        return "Excellent"
    if score >= 70:
        return "Very Good"
    if score >= 60:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    label = None
    error = None

    if request.method == "POST":
        try:
            row = {
                "gender": request.form["gender"],
                "race_ethnicity": request.form["race_ethnicity"],
                "parental_level_of_education": request.form["parental_level_of_education"],
                "lunch": request.form["lunch"],
                "test_preparation_course": request.form["test_preparation_course"],
                "reading_score": float(request.form["reading_score"]),
                "writing_score": float(request.form["writing_score"]),
            }

            input_df = pd.DataFrame([row])
            prediction = float(model.predict(input_df)[0])
            prediction = max(0, min(100, prediction))
            label = performance_label(prediction)

        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        label=label,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
