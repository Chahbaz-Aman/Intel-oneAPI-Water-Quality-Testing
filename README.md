# Is your water drinkable?

![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat&logo=appveyor&logo=Jupyter)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chahbaz-aman-intel-oneapi-water-quality-testing-app-fn87o0.streamlit.app/)
[![git](https://badgen.net/badge/icon/git?icon=git&label)](https://git-scm.com)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com)
[![Visual Studio](https://badgen.net/badge/icon/visualstudio?icon=visualstudio&label)](https://visualstudio.microsoft.com)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)

<h2> 
  Entry for <a href="https://www.hackerearth.com/challenges/hackathon/intel-oneapi-hackathon-for-open-innovation/">Intel® oneAPI Hackathon for Open Innovation</a>
</h2>


## Problem Statement:
<i><a href="https://www.hackerearth.com/challenges/hackathon/intel-oneapi-hackathon-for-open-innovation/">
Freshwater is one of our most vital and scarce natural resources, making up just 3% of the earth’s total water volume. It touches nearly every aspect of our daily lives, from drinking, swimming, and bathing to generating food, electricity, and the products we use every day. Access to a safe and sanitary water supply is essential not only to human life, but also to the survival of surrounding ecosystems that are experiencing the effects of droughts, pollution, and rising temperatures.
</a></i>

<i><a href="https://www.hackerearth.com/challenges/hackathon/intel-oneapi-hackathon-for-open-innovation/">
In this track of the hackathon, <em>participants</em> will have the opportunity to apply the oneAPI skills to help global water security and environmental sustainability efforts by predicting whether freshwater is safe to drink and use for the ecosystems that rely on it.
</a></i>

## Our Solution:
A data science aproach to assess water quality from test results. The dataset provided consolidates <strong>2.7Million freshwater test samples</strong>. A model combining XGBoostClassifier, a neural net and logistic regression trained on this dataset achieved a <strong>89.8% accuracy</strong> in assessing drinkability of a water sample. 

While having more data is better, the development time increases drastically with such large file sizes. The implications of that being reduced productivity and a drop in innovative protyping & testing of ideas. <strong>Intel's oneAPI AI & Analytics Toolkit with optimized libraries offer dramatic boost to all the popular data science frameworks. </strong> This solution was <strong>developed on Intel DevCloud to use the host of XPU options and the oneDAL & oneDNN optimized libraries.</strong>

We created a webapp! [View on Streamlit Community Cloud!](https://chahbaz-aman-intel-oneapi-water-quality-testing-app-fn87o0.streamlit.app/)

Instructions to use the app:
* Upload the test results on the app in CSV format - multiple samples can be processed at once
* View the results 
  * <img src = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/tick.jpg?raw=true' style = "height:20px"/> OK / training-label 0
  * <img src = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/warning.jpg?raw=true' style = "height:20px"/> Unclear
  * <img src = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/cross.jpg?raw=true' style = "height:20px"/> NG / training-label 1

Instructions to run the notebooks:
* Clone the repo to Intel Devcloud JupyterHub
* Set environment variables in terminal
* Use pip to install seaborn library in the base kernel
* EDA
  * Run EDA.ipynb using oneAPI base kernel
* Training 
  * Step 1: Run training_nb_TensorFlow.ipynb using tensorflow kernel & training_nb_XGBoost.ipynb using oneAPI base kernel
  * Step 2: Run training_nb_final_model.ipynb using oneAPI base kernel

<center><img src = 'https://github.com/Chahbaz-Aman/datastore/blob/main/Intel-oneAPI/3519f0c9523c78e1267b548e99bb2249.png?raw=true' style='width:30%'/></center>
