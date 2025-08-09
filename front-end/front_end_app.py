import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import requests
import base64
import os

API_URL = "https://churn-prediction-api-9lrl.onrender.com/predict"

logo_path = "assets/logo.jpeg"
if os.path.exists(logo_path):
    encoded_logo = base64.b64encode(open(logo_path, 'rb').read()).decode()
    logo_component = html.Img(src='data:image/jpeg;base64,{}'.format(encoded_logo), style={'width': '200px'})
else:
    logo_component = html.Div("⚠️ Logo not found! Please add 'logo.jpeg' to the 'assets' folder.", style={'color': 'red'})

app = dash.Dash(__name__)

app.layout = html.Div([
    logo_component,
    html.H1('FutureForecast: AI Churn Prediction'),
    html.H3("Predicting customer churn before it happens, using FastAPI and MLOps Pipeline"),

    html.H4("Customer Information"),

    html.Label("Customer ID"),
    dcc.Input(id='customer_id', type='text', value='123456', style={'width': '100%'}),

    html.Label("Gender"),
    dcc.Dropdown(id='gender', options=[
        {'label': 'Male', 'value': 'Male'},
        {'label': 'Female', 'value': 'Female'}
    ], value='Male'),

    html.Label("Senior Citizen"),
    dcc.Dropdown(id='senior_citizen', options=[
        {'label': '0', 'value': 0},
        {'label': '1', 'value': 1}
    ], value=0),

    html.Label("Partner"),
    dcc.Dropdown(id='partner', options=[
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No', 'value': 'No'}
    ], value='Yes'),

    html.Label("Dependents"),
    dcc.Dropdown(id='dependents', options=[
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No', 'value': 'No'}
    ], value='No'),

    html.Label("Tenure (in months)"),
    dcc.Slider(id='tenure', min=0, max=72, step=1, value=24,
               marks={i: str(i) for i in range(0, 73, 12)}),

    html.Label("Phone Service"),
    dcc.Dropdown(id='phone_service', options=[
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No', 'value': 'No'}
    ], value='Yes'),

    html.Label("Multiple Lines"),
    dcc.Dropdown(id='multiple_lines', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No phone service', 'value': 'No phone service'}
    ], value='No'),

    html.Label("Internet Service"),
    dcc.Dropdown(id='internet_service', options=[
        {'label': 'DSL', 'value': 'DSL'},
        {'label': 'Fiber optic', 'value': 'Fiber optic'},
        {'label': 'No', 'value': 'No'}
    ], value='DSL'),

    html.Label("Online Security"),
    dcc.Dropdown(id='online_security', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Online Backup"),
    dcc.Dropdown(id='online_backup', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Device Protection"),
    dcc.Dropdown(id='device_protection', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Tech Support"),
    dcc.Dropdown(id='tech_support', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Streaming TV"),
    dcc.Dropdown(id='streaming_tv', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Streaming Movies"),
    dcc.Dropdown(id='streaming_movies', options=[
        {'label': 'No', 'value': 'No'},
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No internet service', 'value': 'No internet service'}
    ], value='No'),

    html.Label("Contract"),
    dcc.Dropdown(id='contract', options=[
        {'label': 'Month-to-month', 'value': 'Month-to-month'},
        {'label': 'One year', 'value': 'One year'},
        {'label': 'Two year', 'value': 'Two year'}
    ], value='Month-to-month'),

    html.Label("Paperless Billing"),
    dcc.Dropdown(id='paperless_billing', options=[
        {'label': 'Yes', 'value': 'Yes'},
        {'label': 'No', 'value': 'No'}
    ], value='Yes'),

    html.Label("Payment Method"),
    dcc.Dropdown(id='payment_method', options=[
        {'label': 'Electronic check', 'value': 'Electronic check'},
        {'label': 'Mailed check', 'value': 'Mailed check'},
        {'label': 'Bank transfer (automatic)', 'value': 'Bank transfer (automatic)'},
        {'label': 'Credit card (automatic)', 'value': 'Credit card (automatic)'}
    ], value='Electronic check'),

    html.Label("Monthly Charges"),
    dcc.Input(id='monthly_charges', type='number', value=84.85, step=0.10),

    html.Label("Total Charges"),
    dcc.Input(id='total_charges', type='number', value=1990.50, step=0.10),

    html.Br(),
    html.Button('Predict Churn', id='submit_button'),

    html.Hr(),

    html.Div(id='output_div'),

    html.Hr(),
    html.Div("Built by Khoshaba Odeesho", style={'textAlign': 'center', 'marginTop': '20px'})
])


@app.callback(
    Output('output_div', 'children'),
    Input('submit_button', 'n_clicks'),
    State('customer_id', 'value'),
    State('gender', 'value'),
    State('senior_citizen', 'value'),
    State('partner', 'value'),
    State('dependents', 'value'),
    State('tenure', 'value'),
    State('phone_service', 'value'),
    State('multiple_lines', 'value'),
    State('internet_service', 'value'),
    State('online_security', 'value'),
    State('online_backup', 'value'),
    State('device_protection', 'value'),
    State('tech_support', 'value'),
    State('streaming_tv', 'value'),
    State('streaming_movies', 'value'),
    State('contract', 'value'),
    State('paperless_billing', 'value'),
    State('payment_method', 'value'),
    State('monthly_charges', 'value'),
    State('total_charges', 'value')
)
def predict_churn(n_clicks, customer_id, gender, senior_citizen, partner, dependents, tenure, phone_service,
                  multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support,
                  streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges,
                  total_charges):
    if not n_clicks:
        return ""

    data = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            return html.Div([
                html.H4(f"Prediction Result: {result.get('prediction', 'N/A')}"),
                html.Div(f"Churn Risk: {result.get('churn_risk_percent', 'N/A')}%"),
                html.Div(f"Confidence: {result.get('confidence_percent', 'N/A')}%"),
                html.Div(f"Will Churn: {'Yes' if result.get('will_churn', False) else 'No'}"),
                html.Div(f"Risk Level: {result.get('risk_level', 'N/A')}"),
            ])
        else:
            return html.Div(f"Error from API: {response.status_code} - {response.text}", style={'color': 'red'})
    except requests.exceptions.RequestException as e:
        return html.Div(f"Connection error: {e}", style={'color': 'red'})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)