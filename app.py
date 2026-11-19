from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_shopping_mode_pipeline.joblib"

app = Flask(__name__, template_folder="template")
model = joblib.load(MODEL_PATH)

FORM_SECTIONS = [
    {
        "title": "About you",
        "description": "A little context helps personalize the prediction.",
        "fields": [
            {
                "name": "Age",
                "label": "Age",
                "type": "number",
                "min": 16,
                "max": 54,
            },
            {
                "name": "Gender",
                "label": "Gender",
                "options": ["Female", "Male", "Prefer not to say"],
            },
            {
                "name": "Occupation_Status",
                "label": "Occupation status",
                "options": [
                    "Employed full-time",
                    "Employed part-time",
                    "Self-employed",
                    "Student",
                    "Unemployed",
                ],
            },
            {
                "name": "Country_Region",
                "label": "Country or region",
                "options": [
                    "Brazil",
                    "Colombia",
                    "Egypt",
                    "Germany",
                    "India",
                    "Indonesia",
                    "Kenya",
                    "Mexico",
                    "Nigeria",
                    "Pakistan",
                    "Philippines",
                    "Poland",
                    "Slovakia",
                    "South Africa",
                    "Turkiye",
                    "United Kingdom",
                    "United States",
                    "Vietnam",
                ],
            },
            {
                "name": "Monthly_Income_Range",
                "label": "Monthly income range",
                "options": [
                    "Under $1,000",
                    "$1,000-3,000",
                    "$3,000-6,000",
                    "$6,000+",
                ],
            },
        ],
    },
    {
        "title": "Shopping habits",
        "description": "Tell us where and how often you usually shop.",
        "fields": [
            {
                "name": "Shop_Most_Where",
                "label": "Where do you shop most?",
                "options": ["Online only", "In-store only", "Equal mix"],
            },
            {
                "name": "Online_Shopping_Frequency",
                "label": "How often do you shop online?",
                "options": ["Rarely", "Occasionally", "Often", "Always"],
            },
            {
                "name": "Preferred_Platform",
                "label": "Preferred shopping platform",
                "options": [
                    "Brand websites",
                    "Global marketplace (Amazon etc.)",
                    "Local e-commerce",
                    "Physical stores",
                ],
            },
            {
                "name": "Primary_Device",
                "label": "Device used most for online shopping",
                "options": ["Smartphone", "Laptop", "Tablet"],
            },
            {
                "name": "Top_Spending_Category",
                "label": "Top spending category",
                "options": [
                    "Beauty & personal care",
                    "Books",
                    "Clothing",
                    "Electronics",
                    "Groceries",
                    "miscellaneous",
                ],
            },
            {
                "name": "Avg_Monthly_Spend_NonEssentials",
                "label": "Monthly non-essential shopping spend",
                "options": [
                    "Under $50",
                    "$50-150",
                    "$150-300",
                    "$300-500",
                    "$500+",
                ],
            },
            {
                "name": "Preferred_Payment_Method",
                "label": "Preferred payment method",
                "options": ["Cash", "Credit Card", "Debit Card", "Digital wallet"],
            },
            {
                "name": "Uses_BNPL_Installments",
                "label": "Do you use buy-now-pay-later installments?",
                "options": ["No, never", "Yes, occasionally", "Yes, regularly"],
            },
        ],
    },
    {
        "title": "Shopping behavior",
        "description": "Share the habits and influences that shape your purchases.",
        "fields": [
            {
                "name": "Follows_Brands_On_Social",
                "label": "Do you follow brands on social media?",
                "options": ["Yes", "No"],
            },
            {
                "name": "Makes_Shopping_List",
                "label": "How often do you make a shopping list?",
                "options": ["Never", "Sometimes", "Often", "Always"],
            },
            {
                "name": "Return_Frequency",
                "label": "How often do you return purchases?",
                "options": ["Never", "Sometimes", "Often"],
            },
        ],
    },
    {
        "title": "Your ratings",
        "description": "Rate each item from 1 (low/rarely) to its highest value.",
        "fields": [
            {
                "name": "Review_Influence_Score",
                "label": "How much do reviews influence you?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Social_Ads_Influence_Score",
                "label": "How much do social media ads influence you?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Price_Comparison_Frequency",
                "label": "How often do you compare prices?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Discount_Importance_Score",
                "label": "How important are discounts?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Likely_To_Buy_After_Ad_Score",
                "label": "How likely are you to buy after seeing an ad?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Impulse_Purchase_Frequency",
                "label": "How often do you make impulse purchases?",
                "type": "select_number",
                "min": 1,
                "max": 5,
            },
            {
                "name": "Online_Shopping_Satisfaction_Score",
                "label": "How satisfied are you with online shopping?",
                "type": "select_number",
                "min": 1,
                "max": 10,
            },
            {
                "name": "Trust_In_Online_Reviews_Score",
                "label": "How much do you trust online reviews?",
                "type": "select_number",
                "min": 1,
                "max": 10,
            },
            {
                "name": "Regret_After_Purchase_Frequency",
                "label": "How often do you regret a purchase?",
                "type": "select_number",
                "min": 1,
                "max": 4,
            },
        ],
    },
]

FORM_FIELDS = [field for section in FORM_SECTIONS for field in section["fields"]]
MODEL_FEATURES = list(model.feature_names_in_)
if set(MODEL_FEATURES) != {field["name"] for field in FORM_FIELDS} | {"Age_Group"}:
    raise RuntimeError("The prediction form does not match the saved model features.")


def _age_group(age):
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18\u201324"
    if age <= 34:
        return "25\u201334"
    if age <= 44:
        return "35\u201344"
    return "45\u201354"


def _parse_form(form):
    values = {}
    errors = {}

    for field in FORM_FIELDS:
        name = field["name"]
        raw_value = form.get(name, "").strip()
        if not raw_value:
            errors[name] = "Please complete this field."
            continue

        if field.get("type") in ("number", "select_number"):
            try:
                value = int(raw_value)
            except ValueError:
                errors[name] = "Enter a whole number."
                continue
            if not field["min"] <= value <= field["max"]:
                errors[name] = (
                    f"Choose a value from {field['min']} to {field['max']}."
                )
                continue
            values[name] = value
        else:
            if raw_value not in field["options"]:
                errors[name] = "Choose one of the listed options."
                continue
            values[name] = raw_value

    return values, errors


@app.get("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probabilities = []
    errors = {}
    form_values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        values, errors = _parse_form(request.form)
        if not errors:
            values["Age_Group"] = _age_group(values["Age"])
            features = pd.DataFrame(
                [{name: values[name] for name in MODEL_FEATURES}],
                columns=MODEL_FEATURES,
            )
            prediction = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                probabilities = sorted(
                    zip(model.classes_, model.predict_proba(features)[0]),
                    key=lambda item: item[1],
                    reverse=True,
                )
                probabilities = [
                    {"label": label, "percent": round(float(score) * 100)}
                    for label, score in probabilities
                ]

    return render_template(
        "predict.html",
        sections=FORM_SECTIONS,
        form_values=form_values,
        errors=errors,
        prediction=prediction,
        probabilities=probabilities,
    )


if __name__ == "__main__":
    app.run(debug=True)
