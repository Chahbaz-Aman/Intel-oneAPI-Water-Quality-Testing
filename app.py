import streamlit as st
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from tensorflow import keras
import pickle
from utils import COLS, INDIAN_STANDARDS
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

uploaded_file = left.file_uploader("Upload test results in CSV format")


@st.cache_resource
def load_models():
    scaler = pickle.load(open(SCALER, 'rb'))
    logistic_regressor = pickle.load(open(LOGIT, 'rb'))

    suffix = 'tuned_XGB'
    xgboost = xgb.XGBClassifier()
    xgboost.load_model(f'{XGB_PATH}/model_{suffix}.json')

    suffix = 'largeNN'
    json_file = open(f'{TF_PATH}/model_{suffix}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    neural_net = keras.models.model_from_json(loaded_model_json)
    neural_net.load_weights(f'{TF_PATH}/weights_{suffix}.h5')

    return scaler, xgboost, neural_net, logistic_regressor


def preprocess(df: pd.DataFrame() = st.session_state['df']) -> pd.DataFrame():
    df = df.copy()
    df.set_index('Index', inplace=True)

    numeric_columns = list(df.select_dtypes(exclude=['object']).columns)

    df['Month'] = df['Month'].apply(lambda m: MONTHS[m])

    df = pd.get_dummies(df)
    for col in [col for col in COLS if col not in df.columns]:
        df[col] = 0

    for parameter in INDIAN_STANDARDS.keys():
        if len(INDIAN_STANDARDS[parameter]) > 0:
            try:
                df["unacceptable_ind_"+parameter] = df[parameter].apply(lambda test_result: int((test_result > INDIAN_STANDARDS[parameter][-1]) or
                                                                                                (test_result < INDIAN_STANDARDS[parameter][0])))
            except:
                pass

    df['IND_violations'] = sum(
        [df[col] for col in df.columns if 'unacceptable_ind_' in col])

    numeric_columns = [col for col in numeric_columns if col not in [
        'Target', 'Day', 'Time of Day']]

    df[numeric_columns] = scaler.transform(df[numeric_columns])

    df['Time of Day'] = df['Time of Day'].apply(
        lambda v: np.sin(2 * np.pi * v/24))
    df['Day'] = df['Day'].apply(lambda v: np.sin(2 * np.pi * (v-1)/31))
    df['Month'] = df['Month'].apply(lambda v: np.sin(2 * np.pi * (v-1)/12))

    return df


def predict(df):
    xgb_output = xgboost.predict(df.values)
    nn_output = neural_net.predict(df.values)

    temp = pd.DataFrame(xgb_output, columns=['xgb'])
    temp['nn'] = nn_output

    y_pred = logistic_regressor.predict(temp.values)

    return y_pred


def load_test_results():
    df = pd.read_csv(uploaded_file, dtype=DTYPES)
    if set(df.columns) != set(DTYPES.keys()):
        left.write(df.columns)
        left.write(
            [col for col in df.columns if col not in list(DTYPES.keys())])
        left.write("Uh-oh! Wrong file or incomplete report.")
        return None

    df = df[list(DTYPES.keys())]  # rearrange columns if necessary

    if df.isna().sum().max() != 0:
        left.write("File has missing values!")
        return None

    #backup = df.copy()
    #df['Month'] = df['Month'].apply(lambda m: MONTHS[m])
    #df.set_index('Index', inplace=True)
    #df = pd.get_dummies(df)

    return df


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

    model_output_ = predict(
        st.session_state['X_test'][list(st.session_state['df'].Index == st.session_state['sample'])])
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


scaler, xgboost, neural_net, logistic_regressor = load_models()

if uploaded_file is not None:
    #st.session_state['X_test'], st.session_state['df'] = load_test_results()
    st.session_state['df'] = load_test_results()


if st.session_state['df'] is not None:
    st.session_state['X_test'] = preprocess(st.session_state['df'])

    model_output = predict(st.session_state['X_test'])

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
