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
st.markdown('_Developed using Intel oneAPI Toolkit_')

left, right = st.columns(2)
############################################################################

warnings.filterwarnings('ignore')
st.session_state['sample'] = None
st.session_state['df'] = None
st.session_state['X_test'] = None
st.session_state['flag'] = 0
INDIAN_STANDARDS = {'pH': [6.5, 8.5, 8.5],
                    'Iron': [0, 0.3, 0.3],
                    'Nitrate': [0, 45, 45],
                    'Chloride': [0, 250, 1000],
                    'Lead': [0, 0.01, 0.05],
                    'Zinc': [0, 5, 15],
                    'Color': [0, 5, 15],
                    'Turbidity': [0, 1, 5],
                    'Fluoride': [0, 1, 1.5],
                    'Copper': [0, 0.05, 1.5],
                    'Odor': [0, 1, 2],
                    'Sulfate': [0, 200, 400],
                    'Conductivity': [],
                    'Chlorine': [0, 0.2, 1],
                    'Manganese': [0, 0.1, 0.3],
                    'Total Dissolved Solids': [0, 500, 2000],
                    }

uploaded_file = left.file_uploader("Upload test results in CSV format")
model = xgb.XGBClassifier()
model.load_model(f'{XGB_PATH}/model.json')


def load_test_results():
    df = pd.read_csv(uploaded_file, dtype=DTYPES)
    backup = df.copy()
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

    return df, backup

def translate_standard(range_):
    if len(range_) > 0:
        if range_[0] == 0:
            return f"upto {range_[1]} recommended" + (f", max {range_[-1]}" if len(range_) == 3 and range_[-1] != range_[1] else '')
        else:
            return f"{range_[0]} - {range_[1]}" + (f", max {range_[-1]}" if len(range_) == 3 and range_[-1] != range_[1] else '')
    else:
        return "No spec"

def make_report():

    record = st.session_state['df'][st.session_state['df'].Index ==
                                    st.session_state['sample']]

    model_output_ = model.predict(
        st.session_state['X_test'][list(st.session_state['df'].Index == st.session_state['sample'])].values)
    if model_output_[0] == 0:
        model_output_ = f'''<table>
                            <tr>
                                <td class = 'side-header'>Model Output</td>
                                <td><img src = "{TICK}" style = "height:20px"/></td>
                            </tr>
                        </table>
                        '''
    else:
        model_output_ = f'''<table>
                            <tr>
                                <td class = 'side-header'>Model Output</td>
                                <td><img src = "{CROSS}" style = "height:20px"/></td>
                            </tr>
                        </table>
                        '''

    standards_check = {key: [] for key in INDIAN_STANDARDS.keys()}
    for parameter in INDIAN_STANDARDS.keys():
        try:
            standards_check[parameter].append(f'<img src = "{TICK}" style = "height:20px"/>' if
                                              (record[parameter].iloc[0] >= INDIAN_STANDARDS[parameter][0] and
                                               record[parameter].iloc[0] <= INDIAN_STANDARDS[parameter][1])
                                              else f'<img src = "{CROSS}" style = "height:20px"/>')
        except:
            standards_check[parameter].append(
                f'<img src = "{WARNING}" style = "height:20px"/>')

    standards_check = pd.DataFrame(standards_check).T.reset_index()
    standards_check.rename(
        columns={'index': 'Parameter', 0: 'Assessment'}, inplace=True)
    standards_check['Acceptable Range'] = standards_check['Parameter'].apply(
        lambda r: translate_standard(INDIAN_STANDARDS[r]))
    standards_check['Measured'] = standards_check['Parameter'].apply(
        lambda r: record[r].iloc[0])

    TABLE = standards_check[['Parameter', 'Acceptable Range', 'Measured', 'Assessment']].to_html(
        escape=False, index=False, justify='center', border=0, table_id='standards_table')
    right.markdown(
        f'<div class="standards_report">{model_output_}{TABLE}</div>', unsafe_allow_html=True)

    standards_check['Assessment'] = standards_check['Assessment'].apply(
        lambda v: 'OK' if TICK in v else 'X' if CROSS in v else 'Unknown')

    standards_check['Model Output'] = 'Drinkable' if model_output_[
        0] == 0 else 'NOT Drinkable'

    return standards_check


if uploaded_file is not None:
    st.session_state['X_test'], st.session_state['df'] = load_test_results()


if st.session_state['X_test'] is not None:
    model_output = model.predict(st.session_state['X_test'].values)
    results = pd.DataFrame(model_output, columns=[
                           'Drinkability'], index=st.session_state['X_test'].index)
    results = results.reset_index().rename(columns={'Index': 'Sample Code'})

    results['Drinkability'] = results['Drinkability'].apply(
        lambda x: f'<img src = "{TICK}" style = "height:20px"/>' if x == 0 else f'<img src = "{CROSS}" style = "height:20px"/>')

    TABLE = results.to_html(escape=False,
                            index=False,
                            justify='center',
                            border=0,
                            table_id='results_table'
                            )
    left.markdown(f'<div class="results_report">{TABLE}</div>',
                  unsafe_allow_html=True
                  )

    st.session_state['sample'] = right.selectbox(
        'Detailed report by Sample', results['Sample Code'].values)
    file = make_report()

    right.write('')
    right.download_button(label='Download Report',
                          data=file.to_csv().encode('utf-8'),
                          mime='text/csv',
                          key='download_button'
                          )
