
import streamlit as st

TF_PATH = 'models/Intel_oneDNN_TF_NeuralNet'

XGB_PATH = 'models/Intel_oneDAL_XGBoost'

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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)
