from pages import B_Train_Model, C_Test_Model, D_Cross_Validation
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

############## Assignment 2 Inputs #########
student_filepath = "datasets/housing_dataset.csv"
grader_filepath = "test_dataframe_file/housing_dataset.csv"
student_dataframe = pd.read_csv(student_filepath)
grader_dataframe = pd.read_csv(grader_filepath)
e_dataframe = pd.read_csv(grader_filepath)
test_metrics = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']



@pytest.fixture
def mock_env_setup():
    mock_predict = Mock(return_value=1)
    mock_model = Mock(predict=mock_predict)
    with patch('D_Deploy_App.st.session_state', {'deploy_model': mock_model}) as mock_session_state:
        yield mock_session_state


@pytest.fixture
def L_model():
    model = B_Train_Model.LinearRegression(learning_rate=0.001, num_iterations=500)
    model.W = np.zeros(3)
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2],[4]])
    model.num_examples = 2
    return model
    

@pytest.fixture
def P_model():
    model = B_Train_Model.PolynomailRegression(degree=3, learning_rate=0.001, num_iterations=500)
    model.W = np.zeros(3)
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    return model

@pytest.fixture
def R_model():
    model = B_Train_Model.RidgeRegression(learning_rate=0.4, num_iterations=100, l2_penalty=0.5)
    # model.W = np.zeros((3)) 
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    model.num_examples = model.X.shape[0]
    model.Y_pred=None
    return model

@pytest.fixture
def LR_model():
    model = B_Train_Model.LassoRegression(learning_rate=0.4, num_iterations=100, l1_penalty=0.5)
    # model.W = np.zeros((3)) 
    model.X = np.array([[2.0, 3.4, 4.8], [8.1, 10.13, 13.1]]) 
    model.Y = np.array([[2], [4]])
    model.num_examples = model.X.shape[0]
    model.num_features = model.X.shape[1]
    model.Y_pred=None
    return model

def Test_data():
    y_true = np.array([3,-0.5,2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    return y_true, y_pred
# Checkpoint 1
@pytest.mark.checkpoint1
def test_split_dataset():
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    e_Y = e_dataframe.loc[:, e_dataframe.columns.isin(['median_house_value'])]
    s_split_train_x, s_split_train_y, s_split_test_x, s_split_test_y = B_Train_Model.split_dataset(
        e_X, e_Y, 30)
    assert s_split_train_x.shape == (14448,16)

# Checkpoint 2 
@pytest.mark.checkpoint2
def test_predict_Linear(L_model):
    L_model.W = np.array([1, 8.10, 10.11, 1.13])
    expected_output = np.array([ -18.33999941,  20.33999941]).reshape(-1,1)

    predictions = L_model.predict(L_model.X)

    np.testing.assert_array_almost_equal(predictions, expected_output,
                                         decimal=6, err_msg="Error in prediction method, debug!!", verbose=True)

# Checkpoint 3 
@pytest.mark.checkpoint3
def test_update_weights_Linear(L_model):
    L_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(L_model.W)
    L_model = L_model.update_weights()
    W = np.array([[ 1.004  ],
                  [ 8.06332],
                  [10.07332],
                  [ 1.09332]])
    assert not np.array_equal(old_weights, L_model.W)
    assert L_model is not None
    assert isinstance(L_model, B_Train_Model.LinearRegression)
    np.testing.assert_array_almost_equal(L_model.W, W, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)

# checkpoint 4 
@pytest.mark.checkpoint4
def test_fit_Linear(L_model):
    np.set_printoptions(precision=15)
    updated_weights = np.array([[1.897466235428521],
                                [0.316886934767022],
                                [0.316886935739613],
                                [0.316886937520929]])
    L_model = L_model.fit(L_model.X, L_model.Y)
    np.testing.assert_array_almost_equal(L_model.W, updated_weights, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 5
@pytest.mark.checkpoint5
def test_get_weights(L_model):
    L_model= L_model.fit(L_model.X,L_model.Y)
    print(L_model.W)
    model_name = 'Multiple Linear Regression'
    features = ['feature1', 'feature2', 'feature3']
    
    expected_out_dict = {
        'Multiple Linear Regression': np.array([[1.897466235428521],
       [0.316886934767022],
       [0.316886935739613],
       [0.316886937520929]]),
        'Polynomial Regression': [],
        'Ridge Regression': []
    }
  
    result = L_model.get_weights(model_name, features)
    np.testing.assert_array_almost_equal(result[model_name], expected_out_dict[model_name],decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 6 Polynomial
@pytest.mark.checkpoint6
def test_fit_Polynomial(P_model):
    np.set_printoptions(precision=10)
    #updated_weights = np.array([ 0.16736985, -0.07086507, 0.12156239, 0.25148682])
    updated_weights = np.array([[1.8974662354],
        [0.0526315776],
        [0.0526315777],
        [0.052631578 ],
        [0.0526315791],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315792],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793],
        [0.0526315793]])

    P_model = P_model.fit(P_model.X, P_model.Y)

    np.testing.assert_array_almost_equal(P_model.W, updated_weights, decimal=6, err_msg = "Oh no! Debug!!!, weights are not updated properly, use the formula correctly", verbose =True)


# Checkpoint 7
@pytest.mark.checkpoint7
def test_predict_Polynomial(P_model):
    expected_predictions = np.array([[0.897466], [ 2.897466]])
    P_model = P_model.fit(P_model.X, P_model.Y)
    predictions = P_model.predict(P_model.X)
    np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=6, err_msg="Oh no! Debug!!!,Prediction error!", verbose=True)


# Checkpoint 8
@pytest.mark.checkpoint8
def test_update_weights_Ridge(R_model):
    #np.set_printoptions(precision=15)
    R_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(R_model.W)
    R_model.update_weights()
    ## code pushing done
    expected_weights = np.array([[  2.8     ],
                  [ -4.951998],
                  [ -2.539998],
                  [-13.315999]]) 
    assert not np.array_equal(old_weights, R_model.W)
    assert R_model is not None
    assert isinstance(R_model, B_Train_Model.RidgeRegression)
    np.testing.assert_array_almost_equal(R_model.W, expected_weights, decimal=6, err_msg="Oh no! Debug!!!,Update weight error!", verbose=True)


# Checkpoint 9
@pytest.mark.checkpoint9
def test_update_weights_Lasso(LR_model):
    #np.set_printoptions(precision=15)
    LR_model.W = np.array([1, 8.10, 10.11, 1.13])
    old_weights = np.copy(LR_model.W)
    LR_model.update_weights()
    ## code pushing done
    expected_weights = np.array([[  2.5     ],
                  [ -6.671998],
                  [ -4.661998],
                  [-13.641999]])    
    assert not np.array_equal(old_weights, LR_model.W)
    assert LR_model is not None
    assert isinstance(LR_model, B_Train_Model.LassoRegression)
    np.testing.assert_array_almost_equal(LR_model.W, expected_weights, decimal=6, err_msg="Oh no! Debug!!!,Update weight error!", verbose=True)

# checkpoint 10
@pytest.mark.checkpoint10
def test_rmse():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.rmse(y_true, y_pred)
    expected_error = 0.6123724356957945
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"

# checkpoint 11   
@pytest.mark.checkpoint11 
def test_mae():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.mae(y_true, y_pred)
    expected_error = 0.5
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"

# checkpoint 12
@pytest.mark.checkpoint12 
def test_r2():
    y_true, y_pred =Test_data() 
    error = C_Test_Model.r2(y_true, y_pred)
    expected_error = 0.9486081370449679
    assert np.isclose(error, expected_error, atol=1e-8), f"Debug!!!,Expected error: {expected_error}, but got: {error}"
    
# checkpoint 13
@pytest.mark.cv_tests
def test_cv_perfect_fit():
    #Create simple linear data: y = 2x
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
    Y = np.array([[2], [4], [6], [8], [10], [12], [14], [16], [18]])
    
    #We choose k=3 so we have 3 folds of 3 samples each
    k = 3
    mean_rmse, std_rmse, scores = D_Cross_Validation.k_fold_cross_validation(X, Y, k=k)
    
    #Since it's a perfect line, LinearRegression should have 0 error
    assert np.isclose(mean_rmse, 0.0, atol=1e-8), f"Expected means RMSE ~ 0, got {mean_rmse}"
    assert np.allclose(scores, 0.0, atol=1e-8), f"Expected all fold scores ~ 0, got {scores}"

# checkpoint 14
@pytest.mark.cv_tests
def test_cv_output_structure():
    #Random data
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(20, 3)  # 20 samples, 3 features
    Y = np.random.rand(20, 1)  # 20 samples, 1 target
    
    k = 5
    mean_rmse, std_rmse, scores = D_Cross_Validation.k_fold_cross_validation(X, Y, k=k)
    
    #1. Check types
    assert isinstance(mean_rmse, (float, np.float64, np.float32)), "Mean RMSE should be a float"
    assert isinstance(std_rmse, (float, np.float64, np.float32)), "Std RMSE should be a float"
    assert isinstance(scores, (list, np.ndarray)), "Scores should be a list or numpy array"
    
    #2. Check length of scores
    assert len(scores) == k, f"Expected {k} scores, got {len(scores)}"
    
    #3. Check consistency of calculations
    #Note: Using small tolerance for floating point comparisons
    calc_mean = np.mean(scores)
    calc_std = np.std(scores)
    
    assert np.isclose(mean_rmse, calc_mean, atol=1e-8), "Returned mean doesn't match computed mean of scores"
    assert np.isclose(std_rmse, calc_std, atol=1e-8), "Returned std doesn't match computed std of scores"