import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, ctx
import anthropic

# ============================================================
# AI CLIENT
# ============================================================
# Requires ANTHROPIC_API_KEY to be set as an environment variable.
# Never hardcode the key in source.
ai_client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

# ============================================================
# LOAD DATA
# ============================================================

columbus_rest = pd.read_csv("/vscode/data/clucknstack_columbus_oh_CS-0142.csv")
columbus_rest["store"] = "Columbus, OH"

austin_rest = pd.read_csv("/vscode/data/clucknstack_austin_tx_CS-0317.csv")
austin_rest["store"] = "Austin, TX"

restaurant = pd.concat([columbus_rest, austin_rest], ignore_index=True)
restaurant["order_date"] = pd.to_datetime(restaurant["order_date"])
restaurant["order_time_dt"] = pd.to_datetime(restaurant["order_time"], format="%H:%M:%S")
restaurant["hour"] = restaurant["order_time_dt"].dt.hour
restaurant["day_of_week"] = restaurant["order_date"].dt.day_name()

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ============================================================
# FORECAST FUNCTION (linear regression, per menu item)
# ============================================================

def build_forecast_results(df):
    """Build forecast results dict for a given subset of data (e.g., one store)."""
    FORECAST_DAYS = 30
    results = {}

    daily = (
        df
        .groupby(["menu_item", "category", "order_date"])
        .size()
        .reset_index(name="units_sold")
    )

    items = daily["menu_item"].unique()

    for item_name in items:
        item_df = daily[daily["menu_item"] == item_name].copy()
        category = item_df["category"].iloc[0]
        item_df = item_df.sort_values("order_date")

        day_one = item_df["order_date"].min()
        item_df["day_num"] = (item_df["order_date"] - day_one).dt.days

        X = item_df["day_num"].values.reshape(-1, 1)
        Y = item_df["units_sold"].values

        model = LinearRegression()
        model.fit(X, Y)

        last_day_num = item_df["day_num"].max()
        future_day_nums = np.arange(last_day_num + 1, last_day_num + 1 + FORECAST_DAYS)
        X_future = future_day_nums.reshape(-1, 1)

        Y_pred = model.predict(X_future)
        Y_pred = np.clip(Y_pred, 0, None)

        last_date = item_df["order_date"].max()
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(FORECAST_DAYS)]

        results[item_name] = {
            "category":      category,
            "hist_dates":    item_df["order_date"].tolist(),
            "hist_sales":    item_df["units_sold"].tolist(),
            "future_dates":  future_dates,
            "future_sales":  Y_pred.tolist(),
            "slope":         model.coef_[0],
            "total_pred":    float(Y_pred.sum()),
            "avg_recent":    float(np.mean(item_df["units_sold"].tail(7))) if len(item_df) >= 1 else 0.0,
        }

    return results

# ============================================================
# FIGURE BUILDER (forecast tab)
# ============================================================

CATEGORY_COLORS = {
    "Chicken": "#88E7F7",
    "Fries":   "#88E7F7",
    "Drink":   "#88E7F7",
    "Burger":  "#88E7F7",
}

def build_figure(results):
    ranked = sorted(results.items(), key=lambda x: x[1]["total_pred"], reverse=True)
    item_names_ranked = [name for name, _ in ranked]

    fig = go.Figure()
    trace_pairs = []

    for idx, item_name in enumerate(item_names_ranked):
        d = results[item_name]
        color = CATEGORY_COLORS.get(d["category"], "#2563eb")
        is_visible = (idx == 0)

        fig.add_trace(go.Scatter(
            x=d["hist_dates"], y=d["hist_sales"],
            mode="lines+markers", name="Actual",
            line=dict(color=color, width=2.5), marker=dict(size=5),
            visible=is_visible, legendgroup=item_name, showlegend=True,
        ))

        connect_dates = [d["hist_dates"][-1]] + d["future_dates"]
        connect_sales = [d["hist_sales"][-1]] + d["future_sales"]

        fig.add_trace(go.Scatter(
            x=connect_dates, y=connect_sales,
            mode="lines+markers", name="Forecast",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            visible=is_visible, legendgroup=item_name, showlegend=True,
        ))

        trace_pairs.append((idx * 2, idx * 2 + 1))

    total_traces = len(item_names_ranked) * 2
    dropdown_buttons = []
    for idx, item_name in enumerate(item_names_ranked):
        d = results[item_name]
        visibility = [False] * total_traces
        t1, t2 = trace_pairs[idx]
        visibility[t1] = True
        visibility[t2] = True

        trend_arrow = (
            "▲ trending up" if d["slope"] > 0.005 else
            "▼ trending down" if d["slope"] < -0.005 else
            "→ stable"
        )

        dropdown_buttons.append(dict(
            label=f"{item_name} ({d['category']})",
            method="update",
            args=[
                {"visible": visibility},
                {"title": {"text": (
                    f"<b>{item_name}</b><br>"
                    f"<span style='font-size:13px'>Category: {d['category']} | "
                    f"{trend_arrow} | "
                    f"30‑day forecast: {d['total_pred']:.0f} units</span>"
                )}}
            ]
        ))

    first_name = item_names_ranked[0]
    first_d = results[first_name]
    trend_arrow = (
        "▲ trending up" if first_d["slope"] > 0.0005 else
        "▼ trending down" if first_d["slope"] < -0.0005 else
        "→ stable"
    )

    initial_title = (
        f"<b>{first_name}</b><br>"
        f"<span style='font-size:13px'>Category: {first_d['category']} | "
        f"{trend_arrow} | "
        f"30‑day forecast: {first_d['total_pred']:.0f} units</span>"
    )

    fig.update_layout(
        title=dict(text=initial_title, font=dict(size=18), x=0.02),
        updatemenus=[dict(
            buttons=dropdown_buttons, direction="down", showactive=True,
            x=0.985, xanchor="right", y=1.12, yanchor="top",
            bgcolor="#D5F2F7", bordercolor="black", font=dict(size=12),
        )],
        annotations=[dict(
            text="<b>Select menu item:</b>", x=0.872, xanchor="right",
            y=1.18, yanchor="top", xref="paper", yref="paper",
            showarrow=False, font=dict(size=12),
        )],
        xaxis=dict(title="Date", showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="Units Sold", showgrid=True, gridcolor="#f3f4f6"),
        plot_bgcolor="white", paper_bgcolor="#F5FAFA",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="right", x=1),
        height=480, margin=dict(t=90, b=60, l=50, r=30),
        font=dict(
            family="Nunito, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=13, color="#04324F",
        )
    )

    return fig

# ============================================================
# AI RECOMMENDATION HELPERS
# ============================================================

def get_declining_items(results, slope_threshold=-0.005, top_n=8):
    """Return the most concerning declining items, ranked by severity (slope * volume)."""
    declining = [
        (name, d) for name, d in results.items()
        if d["slope"] < slope_threshold
    ]
    declining.sort(key=lambda x: x[1]["slope"] * max(x[1]["avg_recent"], 1))
    return declining[:top_n]

def build_ai_prompt(store_label, declining_items):
    if not declining_items:
        return None

    lines = []
    for name, d in declining_items:
        lines.append(
            f"- {name} ({d['category']}): recent avg {d['avg_recent']:.1f} units/day, "
            f"daily trend slope {d['slope']:.3f}, 30-day forecast total {d['total_pred']:.0f} units"
        )
    item_block = "\n".join(lines)

    prompt = f"""You are a restaurant operations analyst for a fast-casual chicken chain called Cluck N Stack.

Store/site in view: {store_label}

The following menu items show a declining sales trend based on a linear regression of daily units sold:

{item_block}

For each item, give a short, concrete recommendation (1-2 sentences) for how the store could respond to the decline (e.g., promotion ideas, bundling, pricing, menu placement, possible quality/ops issues to investigate, seasonality considerations). Then add a brief 2-3 sentence overall summary of the biggest risk and the single highest-priority action.

Format your response in markdown with a header per item (use the item name as a bolded line) followed by the recommendation, and end with an "### Overall Summary" section. Keep the entire response concise and skimmable for a busy store manager."""
    return prompt

def call_ai_for_recommendations(store_label, declining_items):
    prompt = build_ai_prompt(store_label, declining_items)
    if prompt is None:
        return "No items are currently showing a meaningful declining trend for this selection. 🎉"

    message = ai_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [block.text for block in message.content if block.type == "text"]
    return "\n".join(text_parts) if text_parts else "The AI did not return a response. Please try again."

# ============================================================
# CHANNEL PREDICTION MODEL (RandomForestClassifier)
# ============================================================
# Goal: predict order_channel (Dine-In / Drive-Thru / Takeout / Delivery)
# from order-level features. This is a multi-class CLASSIFICATION problem,
# which is a different family of ML from the linear regression forecast
# above (regression predicts a number; classification predicts a category).

CHANNEL_FEATURES = ["store", "hour", "day_of_week", "payment_method",
                     "n_items", "total_spent", "n_categories"]

def build_order_level_data(df):
    """Collapse line-item rows up to one row per order, with engineered features."""
    agg = (
        df.groupby("order_id")
        .agg(
            store=("store", "first"),
            hour=("hour", "first"),
            day_of_week=("day_of_week", "first"),
            payment_method=("payment_method", "first"),
            order_channel=("order_channel", "first"),
            n_items=("quantity", "sum"),
            total_spent=("line_total", "sum"),
            n_categories=("category", pd.Series.nunique),
        )
        .reset_index()
    )
    return agg

def train_channel_model(df):
    """Train a RandomForestClassifier to predict order_channel. Returns the
    fitted model, label/feature encoders, and evaluation artifacts."""
    orders = build_order_level_data(df)

    # One-hot encode categorical features, keep numeric features as-is.
    X = pd.get_dummies(
        orders[CHANNEL_FEATURES],
        columns=["store", "day_of_week", "payment_method"],
        drop_first=False,
    )
    feature_columns = X.columns.tolist()

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(orders["order_channel"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_names = y_encoder.classes_.tolist()

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=True)

    return {
        "model": model,
        "y_encoder": y_encoder,
        "feature_columns": feature_columns,
        "accuracy": acc,
        "confusion_matrix": cm,
        "class_names": class_names,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "orders": orders,
    }

def build_confusion_matrix_figure(cm, class_names):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names,
        colorscale=[[0, "#F5FAFA"], [1, "#04324F"]],
        text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix (test set): predicted vs. actual channel",
        xaxis_title="Predicted channel", yaxis_title="Actual channel",
        plot_bgcolor="white", paper_bgcolor="#F5FAFA",
        height=380, margin=dict(t=50, b=40, l=40, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )
    return fig

def build_importance_figure(importances):
    fig = go.Figure(go.Bar(
        x=importances.values, y=importances.index, orientation="h",
        marker_color="#88E7F7",
    ))
    fig.update_layout(
        title="Which features matter most for predicting order channel?",
        xaxis_title="Relative importance", yaxis_title="",
        plot_bgcolor="white", paper_bgcolor="#F5FAFA",
        height=420, margin=dict(t=50, b=40, l=140, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )
    return fig

# Train once at startup on the full combined dataset.
CHANNEL_MODEL_BUNDLE = train_channel_model(restaurant)

def predict_channel(store, hour, dow, payment_method, n_items, total_spent, n_categories):
    bundle = CHANNEL_MODEL_BUNDLE
    row = {
        "store": store, "hour": hour, "day_of_week": dow,
        "payment_method": payment_method, "n_items": n_items,
        "total_spent": total_spent, "n_categories": n_categories,
    }
    row_df = pd.DataFrame([row])
    row_encoded = pd.get_dummies(row_df, columns=["store", "day_of_week", "payment_method"])
    # Align columns with training features (fill missing dummy columns with 0)
    row_encoded = row_encoded.reindex(columns=bundle["feature_columns"], fill_value=0)

    proba = bundle["model"].predict_proba(row_encoded)[0]
    classes = bundle["y_encoder"].classes_
    pred_label = classes[np.argmax(proba)]
    proba_series = pd.Series(proba, index=classes).sort_values(ascending=False)
    return pred_label, proba_series

# ============================================================
# OPERATIONS DASHBOARD HELPERS
# ============================================================

def build_ops_figures(df):
    """Build the four Operations-tab figures for a given subset of data."""
    # --- Channel mix ---
    channel_counts = df.drop_duplicates("order_id")["order_channel"].value_counts()
    channel_fig = go.Figure(go.Pie(
        labels=channel_counts.index, values=channel_counts.values, hole=0.45,
        marker=dict(colors=["#88E7F7", "#04324F", "#D5F2F7", "#5CB8C7"]),
    ))
    channel_fig.update_layout(
        title="Order Channel Mix", paper_bgcolor="#F5FAFA",
        height=340, margin=dict(t=50, b=20, l=20, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )

    # --- Payment mix ---
    payment_counts = df.drop_duplicates("order_id")["payment_method"].value_counts()
    payment_fig = go.Figure(go.Pie(
        labels=payment_counts.index, values=payment_counts.values, hole=0.45,
        marker=dict(colors=["#88E7F7", "#04324F", "#D5F2F7", "#5CB8C7"]),
    ))
    payment_fig.update_layout(
        title="Payment Method Mix", paper_bgcolor="#F5FAFA",
        height=340, margin=dict(t=50, b=20, l=20, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )

    # --- Day-of-week order volume ---
    orders_unique = df.drop_duplicates("order_id")
    dow_counts = orders_unique["day_of_week"].value_counts().reindex(DOW_ORDER).fillna(0)
    dow_fig = go.Figure(go.Bar(
        x=dow_counts.index, y=dow_counts.values, marker_color="#88E7F7",
    ))
    dow_fig.update_layout(
        title="Orders by Day of Week",
        xaxis_title="", yaxis_title="Number of orders",
        plot_bgcolor="white", paper_bgcolor="#F5FAFA",
        height=340, margin=dict(t=50, b=40, l=40, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )

    # --- Hour x day-of-week heatmap ---
    heat = (
        orders_unique.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="orders")
    )
    pivot = heat.pivot(index="day_of_week", columns="hour", values="orders").reindex(DOW_ORDER)
    pivot = pivot.fillna(0)
    heat_fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#F5FAFA"], [1, "#04324F"]],
    ))
    heat_fig.update_layout(
        title="Order Volume by Hour & Day of Week",
        xaxis_title="Hour of day", yaxis_title="",
        paper_bgcolor="#F5FAFA",
        height=340, margin=dict(t=50, b=40, l=80, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )

    return channel_fig, payment_fig, dow_fig, heat_fig

def build_ops_kpis(df):
    orders_unique = df.drop_duplicates("order_id")
    total_orders = len(orders_unique)
    total_revenue = df["line_total"].sum()
    avg_order_value = df.groupby("order_id")["line_total"].sum().mean()
    top_channel = orders_unique["order_channel"].value_counts().idxmax()

    def kpi_card(label, value):
        return html.Div(className="kpi-card", children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
        ])

    return html.Div(className="kpi-row", children=[
        kpi_card("Total Orders", f"{total_orders:,}"),
        kpi_card("Total Revenue", f"${total_revenue:,.2f}"),
        kpi_card("Avg Order Value", f"${avg_order_value:,.2f}"),
        kpi_card("Top Channel", top_channel),
    ])

# ============================================================
# DASH APP LAYOUT
# ============================================================

app = dash.Dash(__name__)
server = app.server

store_options = [
    {"label": "All Stores", "value": "ALL"},
    {"label": "Columbus, OH", "value": "Columbus, OH"},
    {"label": "Austin, TX", "value": "Austin, TX"},
]

app.layout = html.Div(
    children=[
        html.Div(
            className="app-header",
            children=[
                html.Img(src="/assets/burger.jpg"),
                html.Div(children=[
                    html.Div("CLUCK N STACK", className="app-title"),
                    html.Div("ANALYTICS PORTAL", className="app-title2"),
                ])
            ]
        ),

        dcc.Tabs(
            id="tabs",
            value="tab-forecast",
            className="tab-container",
            children=[
                dcc.Tab(
                    label="Forecast", value="tab-forecast",
                    className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(className="tab-content", children=[
                            html.Div(className="card mb-16", children=[
                                html.Div("Site Filter", className="card-title"),
                                html.Div("Choose which site's forecast to view.", className="card-body"),
                                dcc.Dropdown(
                                    id="store-dropdown", options=store_options, value="ALL",
                                    clearable=False, style={"width": "260px", "marginTop": "8px"}
                                )
                            ]),
                            html.Div(className="graph-container", children=[
                                dcc.Graph(id="forecast-graph")
                            ])
                        ])
                    ]
                ),

                dcc.Tab(
                    label="AI Insights", value="tab-ai",
                    className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(className="tab-content", children=[
                            html.Div(className="card mb-16", children=[
                                html.Div("Declining Item Recommendations", className="card-title"),
                                html.Div(
                                    "Uses the store filter from the Forecast tab. Click below to have "
                                    "AI review items with a declining sales trend and suggest actions.",
                                    className="card-body"
                                ),
                                html.Button(
                                    "Generate Recommendations",
                                    id="ai-generate-btn",
                                    n_clicks=0,
                                    style={"marginTop": "12px"}
                                ),
                            ]),
                            dcc.Loading(
                                type="circle",
                                children=html.Div(
                                    id="ai-recommendations-output",
                                    className="card",
                                    style={"whiteSpace": "normal"}
                                )
                            )
                        ])
                    ]
                ),

                dcc.Tab(
                    label="Operations", value="tab-ops",
                    className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(className="tab-content", children=[
                            html.Div(className="card mb-16", children=[
                                html.Div("Site Filter", className="card-title"),
                                html.Div("Choose which site's operations data to view.", className="card-body"),
                                dcc.Dropdown(
                                    id="ops-store-dropdown", options=store_options, value="ALL",
                                    clearable=False, style={"width": "260px", "marginTop": "8px"}
                                )
                            ]),
                            html.Div(id="ops-kpis"),
                            html.Div(className="ops-grid", children=[
                                html.Div(className="card", children=[dcc.Graph(id="ops-channel-fig")]),
                                html.Div(className="card", children=[dcc.Graph(id="ops-payment-fig")]),
                                html.Div(className="card", children=[dcc.Graph(id="ops-dow-fig")]),
                                html.Div(className="card", children=[dcc.Graph(id="ops-heatmap-fig")]),
                            ]),
                        ])
                    ]
                ),

                dcc.Tab(
                    label="Channel Predictor", value="tab-ml",
                    className="tab", selected_className="tab--selected",
                    children=[
                        html.Div(className="tab-content", children=[
                            html.Div(className="card mb-16", children=[
                                html.Div("How Customers Order: a Classification Model", className="card-title"),
                                html.Div(
                                    f"A Random Forest classifier was trained on {CHANNEL_MODEL_BUNDLE['n_train']} "
                                    f"past orders to predict which channel (Dine-In, Drive-Thru, Takeout, or "
                                    f"Delivery) a new order is likely to come through, based on the time, "
                                    f"payment method, store, and basket size. Test-set accuracy: "
                                    f"{CHANNEL_MODEL_BUNDLE['accuracy']*100:.1f}% "
                                    f"(evaluated on {CHANNEL_MODEL_BUNDLE['n_test']} held-out orders, "
                                    f"vs. a random-guess baseline of 25% across 4 classes).",
                                    className="card-body"
                                ),
                            ]),
                            html.Div(className="ops-grid", children=[
                                html.Div(className="card", children=[
                                    dcc.Graph(figure=build_importance_figure(CHANNEL_MODEL_BUNDLE["importances"]))
                                ]),
                                html.Div(className="card", children=[
                                    dcc.Graph(figure=build_confusion_matrix_figure(
                                        CHANNEL_MODEL_BUNDLE["confusion_matrix"],
                                        CHANNEL_MODEL_BUNDLE["class_names"]
                                    ))
                                ]),
                            ]),
                            html.Div(className="card mb-16", children=[
                                html.Div("Try a Prediction", className="card-title"),
                                html.Div(
                                    "Fill in a hypothetical order and the model will predict the most "
                                    "likely channel.",
                                    className="card-body"
                                ),
                                html.Div(className="predict-form", children=[
                                    html.Div(className="form-field", children=[
                                        html.Label("Store"),
                                        dcc.Dropdown(
                                            id="pred-store",
                                            options=[{"label": s, "value": s} for s in ["Columbus, OH", "Austin, TX"]],
                                            value="Columbus, OH", clearable=False,
                                        ),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Day of week"),
                                        dcc.Dropdown(
                                            id="pred-dow",
                                            options=[{"label": d, "value": d} for d in DOW_ORDER],
                                            value="Friday", clearable=False,
                                        ),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Hour of day (0-23)"),
                                        dcc.Input(id="pred-hour", type="number", min=0, max=23, value=18),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Payment method"),
                                        dcc.Dropdown(
                                            id="pred-payment",
                                            options=[{"label": p, "value": p} for p in
                                                     restaurant["payment_method"].unique()],
                                            value="Credit Card", clearable=False,
                                        ),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Number of items in order"),
                                        dcc.Input(id="pred-nitems", type="number", min=1, max=20, value=3),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Total spent ($)"),
                                        dcc.Input(id="pred-total", type="number", min=0, value=12.50, step=0.5),
                                    ]),
                                    html.Div(className="form-field", children=[
                                        html.Label("Number of distinct categories"),
                                        dcc.Input(id="pred-ncat", type="number", min=1, max=4, value=2),
                                    ]),
                                ]),
                                html.Button("Predict Channel", id="predict-btn", n_clicks=0,
                                            style={"marginTop": "12px"}),
                                html.Div(id="predict-output", className="mb-16", style={"marginTop": "16px"}),
                                dcc.Graph(id="predict-proba-fig"),
                            ]),
                        ])
                    ]
                ),
            ]
        )
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

@app.callback(
    Output("forecast-graph", "figure"),
    Input("store-dropdown", "value")
)
def update_forecast(store_value):
    if store_value == "ALL":
        df = restaurant
    else:
        df = restaurant[restaurant["store"] == store_value]

    results = build_forecast_results(df)
    fig = build_figure(results)
    return fig


@app.callback(
    Output("ai-recommendations-output", "children"),
    Input("ai-generate-btn", "n_clicks"),
    State("store-dropdown", "value"),
    prevent_initial_call=True
)
def generate_ai_recommendations(n_clicks, store_value):
    if store_value == "ALL":
        df = restaurant
        store_label = "All Stores"
    else:
        df = restaurant[restaurant["store"] == store_value]
        store_label = store_value

    results = build_forecast_results(df)
    declining_items = get_declining_items(results)

    try:
        ai_text = call_ai_for_recommendations(store_label, declining_items)
    except Exception as e:
        return html.Div([
            html.Div("Something went wrong calling the AI service.", className="card-title"),
            html.Pre(str(e))
        ])

    return dcc.Markdown(ai_text)


@app.callback(
    Output("ops-kpis", "children"),
    Output("ops-channel-fig", "figure"),
    Output("ops-payment-fig", "figure"),
    Output("ops-dow-fig", "figure"),
    Output("ops-heatmap-fig", "figure"),
    Input("ops-store-dropdown", "value"),
)
def update_ops_dashboard(store_value):
    if store_value == "ALL":
        df = restaurant
    else:
        df = restaurant[restaurant["store"] == store_value]

    kpis = build_ops_kpis(df)
    channel_fig, payment_fig, dow_fig, heat_fig = build_ops_figures(df)
    return kpis, channel_fig, payment_fig, dow_fig, heat_fig


@app.callback(
    Output("predict-output", "children"),
    Output("predict-proba-fig", "figure"),
    Input("predict-btn", "n_clicks"),
    State("pred-store", "value"),
    State("pred-dow", "value"),
    State("pred-hour", "value"),
    State("pred-payment", "value"),
    State("pred-nitems", "value"),
    State("pred-total", "value"),
    State("pred-ncat", "value"),
    prevent_initial_call=True
)
def run_prediction(n_clicks, store, dow, hour, payment, n_items, total_spent, n_categories):
    pred_label, proba_series = predict_channel(
        store, hour, dow, payment, n_items, total_spent, n_categories
    )

    message = html.Div([
        html.Span("Predicted channel: ", style={"fontWeight": "600"}),
        html.Span(pred_label, style={"fontWeight": "800", "fontSize": "20px", "color": "#04324F"}),
    ])

    proba_fig = go.Figure(go.Bar(
        x=proba_series.values, y=proba_series.index, orientation="h",
        marker_color="#88E7F7",
    ))
    proba_fig.update_layout(
        title="Predicted probability by channel",
        xaxis_title="Probability", yaxis_title="",
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white", paper_bgcolor="#F5FAFA",
        height=300, margin=dict(t=50, b=40, l=100, r=20),
        font=dict(family="Nunito, sans-serif", size=12, color="#04324F"),
    )

    return message, proba_fig

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))