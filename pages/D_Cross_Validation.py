import numpy as np
import pandas as pd
import streamlit as st
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - Predicting Housing Prices Using Regression")

#############################################

st.title('Cross Validation')

#############################################

# Checkpoints 13 (Extra Credit)
def k_fold_cross_validation(X, Y, k=3, random_state=None):
    """
    Performs k-fold cross validation manually using NumPy.

    Input:
        - X: Features (numpy array)
        - Y: Target (numpy array)
        - k: Number of folds
        - random_state: Random seed for reproducibility (optional)
    Output:
        - mean_rmse: Average RMSE across folds
        - std_rmse: Standard deviation of RMSE across folds
        - scores: List of RMSE scores for each fold
    """
    mean_scores=[]
    std_scores=[]
    scores=[]
    # Add code here
    return mean_scores, std_scores, scores

# Helper functions
def load_dataset(filepath):
    '''
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    '''
    try:
        data = pd.read_csv(filepath)
        st.session_state['house_df'] = data
    except ValueError as err:
            st.write({str(err)})
    return data

random.seed(10)
###################### FETCH DATASET #######################
df = None
filepath = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
if(filepath):
    df = load_dataset(filepath)

if('house_df' in st.session_state):
    df = st.session_state['house_df']

###################### DRIVER CODE #######################

if df is not None:
    # Display dataframe as table
    st.dataframe(df.describe())

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.select_dtypes(include='number').columns),
        key='feature_selectbox',
        index=8
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    feature_input_select = st.multiselect(
        label='Select features for regression input',
        options=[f for f in list(df.select_dtypes(
            include='number').columns) if f != feature_predict_select],
        key='feature_multiselect'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = df.dropna()
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]
    
    # Convert to numpy arrays for CV
    X_np = np.asarray(X.values)
    Y_np = np.asarray(Y.values)

    st.markdown('## Cross Validation Configuration')
    
    st.markdown("Assess the stability of **Multiple Linear Regression** using K-Fold Cross Validation.")
    
    # CV Settings
    k_folds = st.number_input(
        label='Number of k-Folds (1-5)', 
        min_value=2, 
        max_value=5, 
        value=3, 
        step=1,
        help="Select a small number of folds (e.g., 2-5)."
    )
    
    if st.button('Run Cross Validation'):
        if len(feature_input_select) == 0:
            st.error("Please select at least one input feature.")
        else:
            mean_rmse, std_rmse, scores = k_fold_cross_validation(X_np, Y_np, k=k_folds)
            
            st.markdown("### Cross Validation Results")
            st.write(f"**Mean RMSE:** {mean_rmse:.4f} +/- {std_rmse:.4f}")
            
            st.markdown("#### Scores per Fold")
            scores_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(scores))],
                "RMSE": scores
            })
            st.dataframe(scores_df)
            
            st.bar_chart(scores_df.set_index("Fold"))
