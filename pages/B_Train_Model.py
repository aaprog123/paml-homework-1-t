import numpy as np                
import pandas as pd               
from sklearn.model_selection import train_test_split
import streamlit as st             
import random
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - Predicting Housing Prices Using Regression")

#############################################

st.title('Train Model')

#############################################

# Checkpoint 1
def split_dataset(X, y, number,random_state=45):
    """
    This function splits the dataset into 4 parts–- feature 
    and target sets for training and validation.

    Input: 
        - X: training features
        - y: training targets
        - number: the ratio of test samples
    Output: 
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    
    # Add code here

    return X_train, X_val, y_train, y_val

class LinearRegression(object) : 
    def __init__(self, learning_rate=0.001, num_iterations=500): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.cost_history=[]

    # Checkpoint 2
    def predict(self, X): 
        '''
        Make a housing price prediction using model weights and input features X
        
        Input
        - X: matrix of column-wise features
        Output
        - prediction: prediction of house price
        '''
        prediction=None
        # Add code here
        return prediction

    # Checkpoint 3
    def update_weights(self):     
        '''
        Update weights of regression model by computing the 
        derivative of the RSS cost function with respect to weights
        
        Input: None
        Output: None
        ''' 
        # Add code here
        return self
    
    # Checkpoint 4
    def fit(self, X, Y): 
        '''
        Use gradient descent to update the weights for self.num_iterations
        
        Input
            - X: Input features X
            - Y: True values of housing prices
        Output: None
        '''
        self.X = X 
        self.Y = Y
        # Add code here
        return self
    
    # Helper function
    def normalize(self, X):
        '''
        Standardize features X by column

        Input: X is input features (column-wise)
        Output: Standardized features by column
        '''
        X_normalized=X
        # Add code here
        return X_normalized
    
    # Checkpoint 5
    def get_weights(self, model_name, features):
        '''
        This function prints the weights of a regression model in out_dict using the model name as a key.

        Input:
            - model_name: (string) name of model
            - features: input features
        Output:
            - out_dict: a dictionary contains the coefficients of the selected models, with the following keys:
            - 'Multiple Linear Regression'
            - 'Polynomial Regression'
            - 'Ridge Regression'
            - 'Lasso Regression'
        '''
        out_dict = {'Multiple Linear Regression': [],
                'Polynomial Regression': [],
                'Ridge Regression': [],
                'Lasso Regression': []}
        # Add code here
        return out_dict

# Multivariate Polynomial Regression
class PolynomialRegression(LinearRegression):
    def __init__(self, degree, learning_rate, num_iterations):
        self.degree = degree

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Helper function
    def transform(self, X):
        '''
        Converts a matrix of features for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n

        Input
            - X:
        Output
            - X_transform:
        '''
        X_transform=[]
        # Add code here
        return X_transform
    
    # Checkpoint 6
    def fit(self, X, Y):
        '''
        Use gradient descent to update the weights for self.num_iterations

        Input:
            - X: Input features X
            - Y: True values of housing prices
        Output: None
        '''
        self.X = X
        self.Y = Y
        # Add code here
        return self
    
    # Checkpoint 7
    def predict(self, X):
        '''
        Make a prediction using coefficients self.W and input features X
        
        Input
        - X: matrix of column-wise features
        Output
        - prediction: prediction of house price
        '''
        prediction=None
        # Add code here
        return prediction

# Backward compatibility alias for typo in original class name
PolynomailRegression = PolynomialRegression

# Ridge Regression
class RidgeRegression(LinearRegression): 
    def __init__(self, learning_rate, num_iterations, l2_penalty): 
        self.l2_penalty = l2_penalty 

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Checkpoint 8
    def update_weights(self):      
        '''
        Update weights of regression model by computing the 
        derivative of the RSS + l2_penalty*w cost function with respect to weights

        Input: None
        Output: None
        '''
        # Add code here
        return self

# Lasso Regression 
class LassoRegression(LinearRegression): 
    def __init__(self, learning_rate, num_iterations, l1_penalty): 
        self.l1_penalty = l1_penalty 

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Checkpoint 9
    def update_weights(self):      
        '''
        Compute the derivative and update model weights 

        Input: None
        Output: None
        '''
        # Add code here
        return self 

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

    # Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    split_number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Compute the percentage of test and training data
    X_train_df, X_val_df, y_train_df, y_val_df = split_dataset(X, Y, split_number)
    if(len(X_train_df)!=0):
        st.session_state['X_train_df'] = X_train_df
        st.session_state['X_val_df'] = X_val_df
        st.session_state['y_train_df'] = y_train_df
        st.session_state['y_val_df'] = y_val_df

        # Convert to numpy arrays
        X = np.asarray(X.values.tolist()) 
        Y = np.asarray(Y.values.tolist()) 
        X_train, X_val, y_train, y_val = split_dataset(X, Y, split_number)
        train_percentage = (len(X_train) / (len(X_train)+len(y_val)))*100
        test_percentage = (len(X_val)) / (len(X_train)+len(y_val))*100

        st.markdown('Training dataset ({1:.2f}%): {0:.2f}'.format(len(X_train),train_percentage))
        st.markdown('Test dataset ({1:.2f}%): {0:.2f}'.format(len(X_val),test_percentage))
        st.markdown('Total number of observations: {0:.2f}'.format(len(X_train)+len(y_val)))
        train_percentage = (len(X_train)+len(y_train) /
                            (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100
        test_percentage = ((len(X_val)+len(y_val)) /
                            (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression', 
                                  'Ridge Regression',
                                  'Lasso Regression']
    # Collect ML Models of interests
    regression_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=regression_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        regression_model_select))

    # Multiple Linear Regression
    if (regression_methods_options[0] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[0])

        # Add parameter options to each regression method
        learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.01',
            key='mr_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(learning_rate_input))

        num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='mr_iter_textinput'
        )
        st.write('You select the following number of iteration value(s): {}'.format(num_iterations_input))

        multiple_reg_params = {
            'num_iterations': [float(val) for val in num_iterations_input.split(',')],
            'alpha': [float(val) for val in learning_rate_input.split(',')]
        }

        if st.button('Train Multiple Linear Regression Model'):
            # Handle errors
            try:
                multi_reg_model = LinearRegression(learning_rate=multiple_reg_params['alpha'][0], 
                                                   num_iterations=int(multiple_reg_params['num_iterations'][0]))
                multi_reg_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[0]] = multi_reg_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[0] not in st.session_state:
            st.write('Multiple Linear Regression Model is untrained')
        else:
            st.write('Multiple Linear Regression Model trained')

    # Polynomial Regression
    if (regression_methods_options[1] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[1])

        poly_degree = st.number_input(
            label='Enter the degree of polynomial',
            min_value=0,
            max_value=1000,
            value=3,
            step=1,
            key='poly_degree_numberinput'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_degree))

        poly_num_iterations_input = st.number_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            key='poly_num_iter'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_num_iterations_input))

        poly_input=[0.001]
        poly_learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='poly_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(poly_learning_rate_input))

        poly_reg_params = {
            'num_iterations': poly_num_iterations_input,
            'alphas': [float(val) for val in poly_learning_rate_input.split(',')],
            'degree' : poly_degree
        }

        if st.button('Train Polynomial Regression Model'):
            # Handle errors
            try:
                poly_reg_model = PolynomialRegression(poly_reg_params['degree'], 
                                                      poly_reg_params['alphas'][0], 
                                                      poly_reg_params['num_iterations'])
                poly_reg_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[1]] = poly_reg_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[1] not in st.session_state:
            st.write('Polynomial Regression Model is untrained')
        else:
            st.write('Polynomial Regression Model trained')

    # Ridge Regression
    if (regression_methods_options[2] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[2])

        # Add parameter options to each regression method
        ridge_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.01',
            key='ridge_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(ridge_alphas))

        ridge_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='ridge_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(ridge_num_iterations_input))

        ridge_l2_penalty_input = st.text_input(
            label='Enter the l2 penalty (0-1)👇',
            value='1',
            key='ridge_l2_penalty_textinput'
        )
        st.write('You select the following l2 penalty value: {}'.format(ridge_l2_penalty_input))

        ridge_params = {
            'num_iterations': [int(val) for val in ridge_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in ridge_alphas.split(',')],
            'l2_penalty':[float(val) for val in ridge_l2_penalty_input.split(',')]
        }
        if st.button('Train Ridge Regression Model'):
            # Train ridge on all feature --> feature selection
            # Handle Errors
            try:
                ridge_model = RidgeRegression(learning_rate=ridge_params['learning_rate'][0],
                                           num_iterations=ridge_params['num_iterations'][0],
                                           l2_penalty=ridge_params['l2_penalty'][0])
                ridge_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[2]] = ridge_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[2] not in st.session_state:
            st.write('Ridge Model is untrained')
        else:
            st.write('Ridge Model trained')

    # Lasso Regression
    if (regression_methods_options[3] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[3])

        # Add parameter options to each regression method
        lasso_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='lasso_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(lasso_alphas))

        lasso_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='lasso_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(lasso_num_iterations_input))

        lasso_l1_penalty_input = st.text_input(
            label='Enter the l1 penalty (0-1)👇',
            value='0.5',
            key='lasso_l1_penalty_textinput'
        )
        st.write('You select the following l1 penalty value: {}'.format(lasso_l1_penalty_input))

        lasso_params = {
            'num_iterations': [int(val) for val in lasso_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in lasso_alphas.split(',')],
            'l1_penalty':[float(val) for val in lasso_l1_penalty_input.split(',')]
        }
        if st.button('Train Lasso Regression Model'):
            # Train lasso on all feature --> feature selection
            # Handle Errors
            try:
                lasso_model = LassoRegression(learning_rate=lasso_params['learning_rate'][0],
                                                num_iterations=lasso_params['num_iterations'][0],
                                                l1_penalty=lasso_params['l1_penalty'][0])
                lasso_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[3]] = lasso_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[3] not in st.session_state:
            st.write('Lasso Model is untrained')
        else:
            st.write('Lasso Model trained')

    # Store models
    trained_models={}
    for model_name in regression_model_select:
        if(model_name in st.session_state):
            trained_models[model_name] = st.session_state[model_name]

    # Inspect Regression coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select model',
        options=regression_model_select,
        key='inspect_multiselect'
    )
    st.write('You selected the {} models'.format(inspect_models))
    
    models = {}
    weights_dict = {}
    if(inspect_models):
        for model_name in inspect_models:
            weights_dict = trained_models[model_name].get_weights(model_name, feature_input_select)

    # Inspect model cost
    st.markdown('## Inspect model cost')

    # Select multiple models to inspect
    inspect_model_cost = st.selectbox(
        label='Select model',
        options=regression_model_select,
        key='inspect_cost_multiselect'
    )

    st.write('You selected the {} model'.format(inspect_model_cost))

    if(inspect_model_cost):
        try:
            fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True, vertical_spacing=0.1)
            cost_history=trained_models[inspect_model_cost].cost_history

            x_range = st.slider("Select x range:",
                                    value=(0, len(cost_history)))
            st.write("You selected : %d - %d"%(x_range[0],x_range[1]))
            cost_history_tmp = cost_history[x_range[0]:x_range[1]]
            
            fig.add_trace(go.Scatter(x=np.arange(x_range[0],x_range[1],1),
                        y=cost_history_tmp, mode='markers', name=inspect_model_cost), row=1, col=1)

            fig.update_xaxes(title_text="Training Iterations")
            fig.update_yaxes(title_text='Cost', row=1, col=1)
            fig.update_layout(title=inspect_model_cost)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    st.write('Continue to Test Model')