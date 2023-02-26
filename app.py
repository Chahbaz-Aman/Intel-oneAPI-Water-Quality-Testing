import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')


st.title('Is your water drinkable?')

TF_PATH = 'models/Intel_oneDNN_TF_NeuralNet'
XGB_PATH = 'models/Intel_oneDAL_XGBoost'
DTYPES = {'Index': 'int32',
          'pH': 'float32',
          'Iron': 'float32',
          'Nitrate': 'float32',
          'Chloride': 'float32',
          'Lead': 'float64',
          'Zinc': 'float32',
          'Color': 'object',
          'Turbidity': 'float32',
          'Fluoride': 'float32',
          'Copper': 'float32',
          'Odor': 'float32',
          'Sulfate': 'float64',
          'Conductivity': 'float64',
          'Chlorine': 'float64',
          'Manganese': 'float32',
          'Total Dissolved Solids': 'float64',
          'Source': 'object',
          'Water Temperature': 'float64',
          'Air Temperature': 'float64',
          'Month': 'object',
          'Day': 'float16',
          'Time of Day': 'float16',
          }
MONTHS = {'January': 1,
          'February': 2,
          'March': 3,
          'April': 4,
          'May': 5,
          'June': 6,
          'July': 7,
          'August': 8,
          'September': 9,
          'October': 10,
          'November': 11,
          'December': 12,
          }

uploaded_file = st.file_uploader("Upload test results in CSV format")
model = xgb.XGBClassifier()
model.load_model(f'{XGB_PATH}/model.json')
X_test = None


@st.cache_data
def load_test_results():
    df = pd.read_csv(uploaded_file, dtype=DTYPES)
    if set(df.columns) != set(DTYPES.keys()):
        st.write(df.columns)
        st.write([col for col in df.columns if col not in list(DTYPES.keys())])
        st.write("Uh-oh! Wrong file or incomplete results.")
        return None

    df = df[list(DTYPES.keys())]  # rearrange columns if necessary

    if df.isna().sum().max() != 0:
        st.write("File has missing values!")
        return None

    df['Month'] = df['Month'].apply(lambda m: MONTHS[m])
    df.set_index('Index', inplace=True)
    df = pd.get_dummies(df)

    return df.values


if uploaded_file is not None:
    X_test = load_test_results()

if X_test is not None:
    model_output = model.predict(X_test)
    st.write(model_output)
