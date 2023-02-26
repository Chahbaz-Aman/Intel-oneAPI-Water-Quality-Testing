import base64
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from utils import *

############################### LAYOUT #####################################

st.set_page_config(
    page_title='water test',
    layout='wide',
    initial_sidebar_state='collapsed',
)

local_css("style.css")

st.title('Is your water drinkable?')

left, right = st.columns(2)
############################################################################

warnings.filterwarnings('ignore')

uploaded_file = left.file_uploader("Upload test results in CSV format")
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

    return df


if uploaded_file is not None:
    X_test = load_test_results()

if X_test is not None:
    model_output = model.predict(X_test.values)
    results = pd.DataFrame(model_output, columns=[
                           'Result'], index=X_test.index)
    results = results.reset_index().rename(columns={'Index': 'Sample Code'})
    #results = results.set_index('Sample Code')
    results['Result'] = results['Result'].apply(
        lambda x: f'<img src = "{TICK}" style = "height:20px"/>' if x == 0 else f'<img src = "{CROSS}" style = "height:20px"/>')
    left.write(results.to_html(escape=False,
                               index=False,
                               justify='center',
                               border=0,
                               table_id='results_table'
                               ),
               unsafe_allow_html=True
               )
