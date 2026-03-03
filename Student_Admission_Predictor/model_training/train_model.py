import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pickle

# Note: Yeh path tumhari local F: drive ka hai, isay make sure karna ke exist karta ho
sys.path.append("F:/only taha pu/nikah website/Student_Admission_Predictor/model_training")
from utils import *

# ==========================================
# PART 1: LOGISTIC REGRESSION (WITHOUT REGULARIZATION)
# ==========================================

# load dataset
X_train, y_train = load_data("ex2data1.txt")

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:", type(X_train))
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:", type(y_train))
print('The shape of X_train is: ' + str(X_train.shape))
print('The shape of y_train is: ' + str(y_train.shape))
print('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
plt.ylabel('Exam 2 score') 
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

value = 0
print(f"sigmoid({value}) = {sigmoid(value)}")
print("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    loss = 0.0
    total_cost = 0.0
    loss_sum = 0.0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j]*X[i][j] 
            z_wb += z_wb_ij
        z_wb += b    
        f_wb = sigmoid(z_wb)
        loss = (-y[i]* np.log(f_wb)) - ((1-y[i]) * np.log(1 - f_wb))
        loss_sum += loss
        
    total_cost = (1/m) * loss_sum
    return total_cost

m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)
print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))

def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    
    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb_ij = w[j]*X[i][j]  
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw_ij = dj_db_i * X[i][j] 
            dj_dw[j] += dj_dw_ij
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_db, dj_dw

# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}')
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}')

# Compute and display cost and gradient with non-zero w and b
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)
print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    J_history = []
    w_history = []    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db                     
        
        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)
            
        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8
iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

plot_decision_boundary(w, b, X_train, y_train)
plt.ylabel('Exam 2 score') 
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)   
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb += w[j]*X[i][j]         
        z_wb += b        
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
    return p

# Test predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5
tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# Compute accuracy on training set
p = predict(X_train, w, b)
print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))

# ==========================================
# PART 2: REGULARIZED LOGISTIC REGRESSION
# ==========================================

# load dataset 2
X_train, y_train = load_data("ex2data2.txt")

print("X_train:", X_train[:5])
print("Type of X_train:", type(X_train))
print("y_train:", y_train[:5])
print("Type of y_train:", type(y_train))
print('The shape of X_train is: ' + str(X_train.shape))
print('The shape of y_train is: ' + str(y_train.shape))
print('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Microchip Test 2') 
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

print("Original shape of data:", X_train.shape)
mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape    
    cost_without_reg = compute_cost(X, y, w, b)     
    reg_cost = 0.
    for j in range(n):
        reg_cost_j = w[j]**2
        reg_cost = reg_cost + reg_cost_j
    reg_cost = (lambda_ / (2 * m)) * reg_cost    
    total_cost = cost_without_reg + reg_cost
    return total_cost

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
print("Regularized cost :", cost)

def compute_gradient_reg(X, y, w, b, lambda_=1): 
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
        dj_dw_j_reg = (lambda_/m) * w[j]
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg
    return dj_db, dj_dw

np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}")
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}")

# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.
lambda_ = 0.01    
iterations = 10000
alpha = 0.01

w, b, J_history, _ = gradient_descent(X_mapped, y_train, initial_w, initial_b, compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)

plot_decision_boundary(w, b, X_mapped, y_train)
plt.ylabel('Microchip Test 2') 
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

# Compute accuracy on the training set
p = predict(X_mapped, w, b)
print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))

# ==========================================
# PART 3: SAVING THE MODEL
# ==========================================

# NOTE: The notebook used `model` but it was undefined. 
# Saving the learned weights `w` and `b` as a dictionary so it works flawlessly.
model = {"weights": w, "bias": b}

try:
    with open("../backend/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model Saved Successfully")
except Exception as e:
    print(f"Could not save model due to path issue: {e}\nEnsure the '../backend/' directory exists.")