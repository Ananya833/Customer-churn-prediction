from pathlib import Path

import joblib
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET = "Churn"


pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

data = pd.read_csv(PROJECT_ROOT / "Datasets" / "Telco-Customer-Churn_Cleaned.csv")
data["SeniorCitizen"] = data["SeniorCitizen"].astype("category")
bundle = joblib.load(PROJECT_ROOT / "selected_model.joblib")
model = bundle["pipeline"]
features = bundle["features"]
model_name = bundle["model_name"]
metrics = bundle["test_metrics"]
risk_data = bundle["test_risk"]
importance = bundle["feature_importance"]
predictions = bundle["test_predictions"]

colors = {"retained": "#2F6B8A", "churned": "#E07A5F", "ink": "#1F2D3D"}
chart_layout = {"template": "plotly_white", "font": {"family": "Inter, Segoe UI, sans-serif", "color": colors["ink"]}, "margin": {"l": 30, "r": 20, "t": 55, "b": 35}, "legend": {"orientation": "h", "y": -0.22}}


def card(title, value, detail, accent):
    return pn.pane.HTML(f"<div style='background:#fff;border:1px solid #e3e9ef;border-top:4px solid {accent};border-radius:12px;padding:16px 18px;min-height:105px;box-shadow:0 4px 14px rgba(31,45,61,.04)'><div style='color:#64748b;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em'>{title}</div><div style='color:#1f2d3d;font-size:27px;font-weight:750;margin:7px 0 2px'>{value}</div><div style='color:#64748b;font-size:12px'>{detail}</div></div>", sizing_mode="stretch_width", height=118)


total_customers = len(data)
churned_customers = int(data[TARGET].eq("Yes").sum())
retained_customers = total_customers - churned_customers
churn_rate = churned_customers / total_customers
monthly_revenue = data["MonthlyCharges"].sum()
churned_monthly_revenue = data.loc[data[TARGET].eq("Yes"), "MonthlyCharges"].sum()

overview_cards = pn.GridBox(
    card("Customers", f"{total_customers:,}", "records in the supplied dataset", "#2F6B8A"),
    card("Churned", f"{churned_customers:,}", f"{churn_rate:.1%} of customers", "#E07A5F"),
    card("Retained", f"{retained_customers:,}", f"{1 - churn_rate:.1%} of customers", "#49A078"),
    card("Monthly charges", f"${monthly_revenue:,.0f}", f"${churned_monthly_revenue:,.0f} tied to churned accounts", "#7868A6"),
    ncols=4,
    sizing_mode="stretch_width",
)


def churn_distribution():
    count = data[TARGET].value_counts().reindex(["No", "Yes"]).rename_axis("Churn").reset_index(name="Customers")
    fig = px.pie(count, values="Customers", names="Churn", hole=0.64, color="Churn", color_discrete_map={"No": colors["retained"], "Yes": colors["churned"]})
    fig.update_layout(**chart_layout, title="Customer outcome mix", annotations=[{"text": f"{churn_rate:.1%}<br>churn", "x": .5, "y": .5, "showarrow": False, "font": {"size": 18, "color": colors["ink"]}}])
    return pn.pane.Plotly(fig, height=350, sizing_mode="stretch_width")


def churn_rate_chart(column, title, horizontal=False):
    rates = data.groupby(column, observed=True)[TARGET].apply(lambda values: values.eq("Yes").mean()).rename("Churn rate").reset_index()
    if horizontal:
        fig = px.bar(rates.sort_values("Churn rate"), x="Churn rate", y=column, orientation="h", color="Churn rate", color_continuous_scale=["#a9c7d6", "#e07a5f"])
    else:
        fig = px.bar(rates, x=column, y="Churn rate", color="Churn rate", color_continuous_scale=["#a9c7d6", "#e07a5f"])
    fig.update_layout(**chart_layout, title=title, coloraxis_showscale=False, yaxis_tickformat=".0%" if not horizontal else None, xaxis_tickformat=".0%" if horizontal else None)
    return pn.pane.Plotly(fig, height=360, sizing_mode="stretch_width")


tenure_data = data.assign(TenureGroup=pd.cut(data["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"]))
tenure_rates = tenure_data.groupby("TenureGroup", observed=True)[TARGET].apply(lambda values: values.eq("Yes").mean()).rename("Churn rate").reset_index()
tenure_fig = px.bar(tenure_rates, x="TenureGroup", y="Churn rate", color="Churn rate", color_continuous_scale=["#a9c7d6", "#e07a5f"])
tenure_fig.update_layout(**chart_layout, title="Churn by tenure group", coloraxis_showscale=False, yaxis_tickformat=".0%")

payment_chart = churn_rate_chart("PaymentMethod", "Churn by payment method", True)
contract_chart = churn_rate_chart("Contract", "Churn by contract type")
internet_chart = churn_rate_chart("InternetService", "Churn by internet service")
tenure_chart = pn.pane.Plotly(tenure_fig, height=360, sizing_mode="stretch_width")

model_metrics_fig = px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Metric", y="Score", color="Model", barmode="group", color_discrete_sequence=["#2F6B8A", "#E07A5F", "#49A078"])
model_metrics_fig.update_layout(**chart_layout, title="Held-out model comparison", yaxis_range=[0, 1], yaxis_tickformat=".0%")
model_metrics_chart = pn.pane.Plotly(model_metrics_fig, height=390, sizing_mode="stretch_width")

confusion = pd.crosstab(predictions["actual"], predictions["predicted"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
cm_fig = go.Figure(data=go.Heatmap(z=confusion.values, x=["Retained", "Churned"], y=["Retained", "Churned"], colorscale=[[0, "#eaf1f5"], [1, "#2F6B8A"]], text=confusion.values, texttemplate="%{text}", showscale=False))
cm_fig.update_layout(**chart_layout, title=f"Confusion matrix · {model_name}", xaxis_title="Predicted class", yaxis_title="Actual class", height=390)
importance_fig = px.bar(importance.head(12).sort_values("importance"), x="importance", y="feature_group", orientation="h", color="importance", color_continuous_scale="Teal")
importance_fig.update_layout(**chart_layout, title="Most influential feature groups", coloraxis_showscale=False, xaxis_title="Relative importance", yaxis_title="")

contract_filter = pn.widgets.Select(name="Contract", options=["All"] + sorted(risk_data["Contract"].dropna().unique().tolist()), value="All")
risk_filter = pn.widgets.Select(name="Estimated risk", options=["All", "Higher", "Medium", "Lower"], value="All")
payment_filter = pn.widgets.Select(name="Payment method", options=["All"] + sorted(data["PaymentMethod"].dropna().unique().tolist()), value="All")


def filtered_risk(contract, risk_band, payment_method):
    filtered = risk_data.copy()
    if contract != "All":
        filtered = filtered[filtered["Contract"] == contract]
    if risk_band != "All":
        filtered = filtered[filtered["risk_band"] == risk_band]
    if payment_method != "All":
        ids = data.loc[data["PaymentMethod"] == payment_method, "customerID"]
        filtered = filtered[filtered["customerID"].isin(ids)]
    return filtered.sort_values("churn_probability", ascending=False).head(100)


risk_table = pn.widgets.Tabulator(pd.DataFrame(), pagination="local", page_size=12, height=390, sizing_mode="stretch_width", formatters={"churn_probability": {"type": "progress", "max": 1}})


def update_risk_table(event=None):
    risk_table.value = filtered_risk(contract_filter.value, risk_filter.value, payment_filter.value)


for widget in [contract_filter, risk_filter, payment_filter]:
    widget.param.watch(update_risk_table, "value")
update_risk_table()

input_widgets = {
    "gender": pn.widgets.Select(name="Gender", options=["Female", "Male"], value="Female"),
    "SeniorCitizen": pn.widgets.Select(name="Senior citizen", options=[0, 1], value=0),
    "Partner": pn.widgets.Select(name="Partner", options=["Yes", "No"], value="Yes"),
    "Dependents": pn.widgets.Select(name="Dependents", options=["Yes", "No"], value="No"),
    "tenure": pn.widgets.IntSlider(name="Tenure (months)", start=0, end=72, value=12),
    "PhoneService": pn.widgets.Select(name="Phone service", options=["Yes", "No"], value="Yes"),
    "MultipleLines": pn.widgets.Select(name="Multiple lines", options=["No", "Yes", "No phone service"], value="No"),
    "InternetService": pn.widgets.Select(name="Internet service", options=["DSL", "Fiber optic", "No"], value="Fiber optic"),
    "OnlineSecurity": pn.widgets.Select(name="Online security", options=["Yes", "No", "No internet service"], value="No"),
    "OnlineBackup": pn.widgets.Select(name="Online backup", options=["Yes", "No", "No internet service"], value="No"),
    "DeviceProtection": pn.widgets.Select(name="Device protection", options=["Yes", "No", "No internet service"], value="No"),
    "TechSupport": pn.widgets.Select(name="Tech support", options=["Yes", "No", "No internet service"], value="No"),
    "StreamingTV": pn.widgets.Select(name="Streaming TV", options=["Yes", "No", "No internet service"], value="No"),
    "StreamingMovies": pn.widgets.Select(name="Streaming movies", options=["Yes", "No", "No internet service"], value="No"),
    "Contract": pn.widgets.Select(name="Contract", options=["Month-to-month", "One year", "Two year"], value="Month-to-month"),
    "PaperlessBilling": pn.widgets.Select(name="Paperless billing", options=["Yes", "No"], value="Yes"),
    "PaymentMethod": pn.widgets.Select(name="Payment method", options=sorted(data["PaymentMethod"].unique().tolist()), value="Electronic check"),
    "MonthlyCharges": pn.widgets.FloatInput(name="Monthly charges", value=75.0, start=0, end=200, step=0.5),
    "TotalCharges": pn.widgets.FloatInput(name="Total charges", value=900.0, start=0, end=10000, step=10),
}
predict_button = pn.widgets.Button(name="Estimate churn risk", button_type="primary", icon="chart-line")
prediction_output = pn.pane.Markdown("Enter customer details and select **Estimate churn risk**.", styles={"background": "#f4f7fa", "padding": "18px", "border-radius": "10px"})


def run_prediction(event):
    values = {column: widget.value for column, widget in input_widgets.items()}
    frame = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(frame)[0, 1])
    predicted = int(model.predict(frame)[0])
    label = "Higher" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Lower"
    outcome = "Churn flagged" if predicted == 1 else "Churn not flagged"
    top = ", ".join(importance.head(3)["feature_group"].tolist())
    prediction_output.object = f"### {outcome}\n\n**Estimated churn probability:** {probability:.1%}  ·  **Risk band:** {label}\n\nThis estimate uses the selected {model_name} model. The risk band is a dashboard grouping; the classification threshold is 0.50. The most influential feature groups in the held-out model are {top}. These are global model associations, not a causal explanation for this individual."


predict_button.on_click(run_prediction)

overview = pn.Column("## Business overview", "Historical customer counts and monthly charges in the supplied dataset.", overview_cards, pn.Row(churn_distribution(), contract_chart), pn.Row(tenure_chart, payment_chart), pn.Row(internet_chart), sizing_mode="stretch_width")
eda = pn.Column("## Exploratory analysis", "Churn rates are descriptive group comparisons and do not establish cause.", pn.Row(contract_chart, tenure_chart), pn.Row(payment_chart, internet_chart), sizing_mode="stretch_width")
model_section = pn.Column("## Model comparison", f"Selected by highest F1-score on a validation slice from the training data, with recall and ROC-AUC as tie-breakers. The held-out test split is used only for final comparison. Current selection: **{model_name}**. All models use balanced class weights.", model_metrics_chart, pn.Row(pn.pane.Plotly(cm_fig, height=390, sizing_mode="stretch_width"), pn.pane.Plotly(importance_fig, height=390, sizing_mode="stretch_width")), sizing_mode="stretch_width")
risk_section = pn.Column("## Customer risk analysis", "Estimated probabilities for the held-out test customers only. Use filters to explore the highest estimated risks.", pn.Row(contract_filter, risk_filter, payment_filter), risk_table, sizing_mode="stretch_width")
prediction_section = pn.Column("## Individual customer estimate", "Enter available customer and service details. The output is a classification estimate from historical labels, not a prediction of when a customer will leave.", pn.GridBox(*input_widgets.values(), ncols=3, sizing_mode="stretch_width"), predict_button, prediction_output, sizing_mode="stretch_width")

template = pn.template.FastListTemplate(
    title="Telecom Churn · Retention Intelligence",
    site="CUSTOMER ANALYTICS",
    site_url="",
    accent_base_color="#2F6B8A",
    header_background="#18364A",
    main=[pn.Tabs(("Overview", overview), ("EDA", eda), ("Models", model_section), ("Risk explorer", risk_section), ("Customer estimate", prediction_section), dynamic=True)],
    theme="default",
)

template.servable()
