### Flask App: Customer Segmentation Predictor
# Loads a pre-trained KMeans model + scaler, serves a form, predicts a
# customer's segment from 8 numeric inputs.

import os                                   # imported but never used -- can be removed
import joblib                                # for loading the pickled model/scaler
from flask import Flask, request, render_template

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load both files on startup
try:
    model = joblib.load(MODEL_PATH)          # fitted KMeans object from the notebook
    scaler = joblib.load(SCALER_PATH)         # fitted StandardScaler from the notebook
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    # NOTE: app still starts even if loading failed -- model/scaler stay None,
    # and every request will hit the "Server error" branch below. That's handled
    # gracefully for POST, but there's no /health or startup check that fails loudly.

# Combined Route for handling both page view and form submission
@app.route("/", methods=["GET", "POST"])
def index():
    # Default values for the form fields -- also used to repopulate the form after submit
    form_data = {
        "AnnualIncome": 55000, "TotalSpend": 1200, "PurchaseFrequencyPerMonth": 4,
        "WebsiteVisitsPerMonth": 12, "AppUsageHoursPerMonth": 8,
        "DiscountUsesPerYear": 5, "Returns": 1, "LastPurchaseDays": 14
    }

    # Handle GET request (initial page load)
    if request.method == "GET":
        return render_template("index.html", form_data=form_data, result=None, error=None)

    # Handle POST request (form submission without JavaScript)
    if request.method == "POST":
        if model is None or scaler is None:
            # triggered whenever the try/except above failed at startup
            error_msg = "Server error: Model or Scaler not loaded."
            return render_template("index.html", form_data=form_data, result=None, error=error_msg)

        # Read form inputs and keep track of values to persist them in the UI
        try:
            for key in form_data.keys():
                form_data[key] = float(request.form.get(key, 0))
                # NOTE: no validation on range/sign here -- e.g. negative "Returns"
                # or "DiscountUsesPerYear" would be silently accepted and scaled/predicted

            # Extract features in correct order
            # NOTE: this order MUST exactly match `features` in the notebook's cell 6
            # (AnnualIncome, TotalSpend, PurchaseFrequencyPerMonth, WebsiteVisitsPerMonth,
            #  AppUsageHoursPerMonth, DiscountUsesPerYear, Returns, LastPurchaseDays) --
            # it does match here, but nothing enforces that if either file changes later.
            raw_features = [[
                form_data["AnnualIncome"], form_data["TotalSpend"], form_data["PurchaseFrequencyPerMonth"],
                form_data["WebsiteVisitsPerMonth"], form_data["AppUsageHoursPerMonth"],
                form_data["DiscountUsesPerYear"], form_data["Returns"], form_data["LastPurchaseDays"]
            ]]

            # Machine learning operations
            scaled_features = scaler.transform(raw_features)   # apply the SAME scaling as training
            cluster = model.predict(scaled_features)            # nearest-centroid cluster id
            cluster_id = int(cluster[0])

            # *** BUG: hardcoded mapping only covers cluster ids 0 and 1 ***
            if cluster_id == 0:
                cluster_segment = "Low Engagement"
            elif cluster_id == 1:
                cluster_segment = "VIP Customers"
            # if the model was trained with best_k > 2 (see notebook cell 9), any
            # cluster_id >= 2 falls through both branches, `cluster_segment` is never
            # assigned, and the next line raises NameError -- caught below and shown
            # as a generic, unhelpful error message to the end user.

            result_msg = f"Customer belongs to Segment: {cluster_segment}"
            return render_template("index.html", form_data=form_data, result=result_msg, error=None)

        except Exception as e:
            # catches bad float() conversions AND the NameError above AND anything else --
            # too broad to distinguish user input errors from real bugs
            return render_template("index.html", form_data=form_data, result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # NOTE: debug=True enables the Werkzeug interactive debugger, which allows
    # arbitrary code execution if this is ever exposed beyond localhost -- turn
    # off before deploying anywhere reachable by others.
