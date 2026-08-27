import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")
st.title("💰 Insurance Charges Prediction")
st.caption("Cleans the insurance dataset, encodes it, trains several regression models, and compares them.")

# ---------------------------------------------------------------------
# 1) Load data (bundled with the repo — no upload needed)
# ---------------------------------------------------------------------
DEFAULT_DATA_PATH = "insurance.csv"  # must sit next to app.py in the repo

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df_raw = load_data(DEFAULT_DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_DATA_PATH}` next to app.py in the repo. "
        "Make sure insurance.csv is uploaded to the same GitHub repository as app.py."
    )
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df_raw.head())

# ---------------------------------------------------------------------
# 2) Cleaning
# ---------------------------------------------------------------------
df = df_raw.drop_duplicates().copy()

col1, col2 = st.columns(2)
with col1:
    st.write("**Missing values**")
    st.dataframe(df.isnull().sum())
with col2:
    st.write("**Shape after dropping duplicates**")
    st.write(df.shape)

# ---------------------------------------------------------------------
# 3) Encode categoricals with LabelEncoder (matches notebook v2)
# ---------------------------------------------------------------------
le_sex = LabelEncoder()
le_region = LabelEncoder()
le_smoker = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])          # e.g. male=1, female=0
df['region'] = le_region.fit_transform(df['region'])  # alphabetical order
df['smoker'] = le_smoker.fit_transform(df['smoker'])  # yes=1, no=0

st.subheader("Encoded data preview")
st.dataframe(df.head())
st.caption(
    f"sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))} | "
    f"smoker: {dict(zip(le_smoker.classes_, le_smoker.transform(le_smoker.classes_)))} | "
    f"region: {dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))}"
)

# ---------------------------------------------------------------------
# 4) Correlation heatmap
# ---------------------------------------------------------------------
st.subheader("Correlation heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="cool", linewidths=2, ax=ax)
st.pyplot(fig)

# ---------------------------------------------------------------------
# 5) Boxplots per column
# ---------------------------------------------------------------------
with st.expander("Boxplots for every column"):
    for c in df.columns:
        fig_b, ax_b = plt.subplots(figsize=(6, 2))
        sns.boxplot(x=df[c], ax=ax_b)
        ax_b.set_title(f"Boxplot of {c}")
        st.pyplot(fig_b)
        plt.close(fig_b)

# ---------------------------------------------------------------------
# 6) Distribution of charges by smoker
# ---------------------------------------------------------------------
st.subheader("Distribution of charges by smoker")
fig_d, ax_d = plt.subplots(figsize=(8, 4))
sns.histplot(data=df, x='charges', hue='smoker', kde=True, ax=ax_d)
ax_d.set_xlabel('Charges')
ax_d.set_ylabel('Count')
ax_d.set_title('Distribution of Charges by Smoker')
st.pyplot(fig_d)

# ---------------------------------------------------------------------
# 7) Train/test split + scaling
# ---------------------------------------------------------------------
x = df.drop(['charges'], axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ---------------------------------------------------------------------
# 8) Train models
# ---------------------------------------------------------------------
st.sidebar.header("Models to train")
run_svr = st.sidebar.checkbox("SVR (slow, low accuracy on this data)", value=False)
run_xgb = st.sidebar.checkbox("XGBoost", value=XGB_AVAILABLE, disabled=not XGB_AVAILABLE)

@st.cache_resource
def train_models(x_train_scaled, y_train, run_svr, run_xgb):
    models = {}

    models['Linear'] = LinearRegression().fit(x_train_scaled, y_train)
    models['KNN'] = KNeighborsRegressor(n_neighbors=3).fit(x_train_scaled, y_train)
    models['RandomForestRegressor'] = RandomForestRegressor(
        n_estimators=200, random_state=42
    ).fit(x_train_scaled, y_train)
    models['GradientBoostingRegressor'] = GradientBoostingRegressor().fit(x_train_scaled, y_train)
    models['DecisionTreeRegressor'] = DecisionTreeRegressor(
        max_depth=3, random_state=42
    ).fit(x_train_scaled, y_train)

    if run_svr:
        models['SVR'] = SVR(C=100, epsilon=0.01, gamma=0.1, kernel='rbf').fit(x_train_scaled, y_train)

    if run_xgb and XGB_AVAILABLE:
        xg = XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, subsample=0.8)
        xg.fit(x_train_scaled, y_train)
        models['XGBRegressor'] = xg

    return models

with st.spinner("Training models..."):
    models = train_models(x_train_scaled, y_train, run_svr, run_xgb)

# ---------------------------------------------------------------------
# 9) Evaluate
# ---------------------------------------------------------------------
rows = []
for name, model in models.items():
    y_pred = model.predict(x_test_scaled)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    })

results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

st.subheader("Model comparison")
st.dataframe(results_df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}))

fig2, ax2 = plt.subplots(figsize=(8, 4))
results_df.set_index("Model")["R2"].sort_values().plot(kind="bar", ax=ax2, title="R2 Score by Model")
ax2.set_ylabel("R2")
st.pyplot(fig2)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
st.success(f"🏆 Best model: **{best_model_name}** (R² = {results_df.iloc[0]['R2']:.4f})")

# ---------------------------------------------------------------------
# 10) Live prediction
# ---------------------------------------------------------------------
st.header("Try a prediction")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex_in = st.selectbox("Sex", list(le_sex.classes_))
with c2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
with c3:
    smoker_in = st.selectbox("Smoker", list(le_smoker.classes_))
    region_in = st.selectbox("Region", list(le_region.classes_))

if st.button("Predict charges"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": le_sex.transform([sex_in])[0],
        "bmi": bmi,
        "children": children,
        "smoker": le_smoker.transform([smoker_in])[0],
        "region": le_region.transform([region_in])[0],
    }])[x.columns]  # keep same column order as training data

    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)[0]
    st.metric("Predicted insurance charge", f"${prediction:,.2f}")
