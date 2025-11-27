# app_streamlit.py
with col1:
st.header("Single transaction")
# Build inputs for each feature (only numeric input for Time & Amount; others default 0)
default_vals = {f: 0.0 for f in features}
if "Time" in features:
default_vals["Time"] = 0.0
if "Amount" in features:
default_vals["Amount"] = 1.0


inputs = {}
# show Time and Amount prominently
if "Time" in features:
inputs["Time"] = st.number_input("Time", value=float(default_vals["Time"]))
# show Amount
if "Amount" in features:
inputs["Amount"] = st.number_input("Amount", value=float(default_vals["Amount"]))


# show V1..V28 if present in features
for f in features:
if f in ("Time", "Amount"):
continue
# small input to avoid huge UI
inputs[f] = st.number_input(f, value=float(default_vals.get(f, 0.0)))


if st.button("Predict single transaction"):
try:
df = pd.DataFrame([inputs])[features]
# scale Time & Amount
if "Time" in features and "Amount" in features and scaler is not None:
df[["Time","Amount"]] = scaler.transform(df[["Time","Amount"]])
proba = float(model.predict_proba(df)[:,1][0])
pred = int(model.predict(df)[0])
st.success(f"Prediction: {pred} — fraud probability: {proba:.4f}")
except Exception as e:
st.error(str(e))


with col2:
st.header("Batch predictions (CSV)")
st.markdown("Upload CSV with the same feature columns used for training.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
try:
df = pd.read_csv(uploaded_file)
missing = [c for c in features if c not in df.columns]
if missing:
st.error(f"Missing columns in CSV: {missing}")
else:
X = df[features].copy()
if "Time" in features and "Amount" in features and scaler is not None:
X[["Time","Amount"]] = scaler.transform(X[["Time","Amount"]])
probs = model.predict_proba(X)[:,1]
preds = model.predict(X).astype(int)
out = df.copy()
out["fraud_probability"] = probs
out["prediction"] = preds
st.dataframe(out.head(50))


csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
except Exception as e:
st.error(str(e))


st.markdown("---")
st.markdown("**Notes:** The model expects the same feature columns used during training (features are stored inside the pickle). If you get missing column errors, open the `fraud_model.pkl` details to see the `features` list.")


# Footer
st.write("Model loaded from:", MODEL_PATH)
