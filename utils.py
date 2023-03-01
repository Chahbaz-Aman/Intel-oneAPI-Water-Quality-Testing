import streamlit as st
import pandas as pd

TF_PATH = 'models/Intel_oneDNN_TF_NeuralNet'
XGB_PATH = 'models/Intel_oneDAL_XGBoost'
LOGIT = 'models/logistic.sav'
SCALER = 'models/scaler.pkl'

CROSS = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/cross.jpg?raw=true'
TICK = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/tick.jpg?raw=true'
WARNING = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/warning.jpg?raw=true'

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

COLS = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Turbidity',
       'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine',
       'Manganese', 'Total Dissolved Solids', 'Water Temperature',
       'Air Temperature', 'Month', 'Day', 'Time of Day',
       'Color_Colorless', 'Color_Faint Yellow', 'Color_Light Yellow',
       'Color_Near Colorless', 'Color_Yellow', 'Source_Aquifer',
       'Source_Ground', 'Source_Lake', 'Source_Reservoir', 'Source_River',
       'Source_Spring', 'Source_Stream', 'Source_Well', 'unacceptable_ind_pH',
       'unacceptable_ind_Iron', 'unacceptable_ind_Nitrate',
       'unacceptable_ind_Chloride', 'unacceptable_ind_Lead',
       'unacceptable_ind_Zinc', 'unacceptable_ind_Turbidity',
       'unacceptable_ind_Fluoride', 'unacceptable_ind_Copper',
       'unacceptable_ind_Odor', 'unacceptable_ind_Sulfate',
       'unacceptable_ind_Chlorine', 'unacceptable_ind_Manganese',
       'unacceptable_ind_Total Dissolved Solids', 'IND_violations']

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)
