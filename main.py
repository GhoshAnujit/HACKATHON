import streamlit as st
import numpy as np
import pickle

import xgboost as xgb

#filename=
st.markdown("""# Life Expectancy Predictor!""")
progress = st.progress(0)
model = xgb.XGBRegressor()
model.load_model("xgboost/final_model.txt")
features = []
#{'nthread': 4}
features.append(st.text_input(label = "Adult Mortality" , value = 0.5))
features.append(st.text_input(label = "Schooling" , value = 100.0))
features.append(st.text_input(label = "Diphtheria" , value = 10.0))
features.append(st.text_input(label = "Percentage Expenditure" , value = 0.5))
features.append(st.text_input(label = "Infant Deaths" , value = 500.0))
features.append(st.text_input(label = "GDP" , value = 1000.0))
features.append(st.text_input(label = "Income component of resources" , value = 5.0))
features.append(st.text_input(label = "Total Expenditure" , value = 5.0))
features.append(st.text_input(label = "malnutrition 5-9 years" , value = 5.0))
features.append(st.text_input(label = "malnutrition 1-19 years" , value = 5.0))

print(model.get_booster().feature_names)
X = np.array(features , dtype=float)

progress.progress(100)
if st.button("Submit"):
    prediction = model.predict([X])
    st.text(prediction[0])
