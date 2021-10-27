'''
    File name: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020 - Perceptron Model Done
                       10/30/2020 - Linear Regression 2D
    Python Version: 3.7

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

############################### Section-B Linear Regression #####################################################

class Linear_Regression:
	#Initialize Instance variables and function
	def __init__(self, LR_learning_rate, LR_Iteration):
		self.LR_learning_rate = LR_learning_rate
		self.LR_Iteration = LR_Iteration
		self.init_b = 0
		self.init_m = 0
		self.compute_error = self.compute_error
		self.gradient_descent_runner = self.gradient_descent_runner
		self.step_gradient_calculate = self.step_gradient_calculate
		self.plot_graph = self.plot_graph 
		self.plot_gradient_Curve = self.plot_gradient_Curve
		self.X = 0
		self.Y = 0

	def plot_graph(self, points_xy, m, b):
	    #Display Final Output with Points and Regression Line
	    m_trim = np.round(m, 2)
	    b_trim = np.round(b, 2)
	   #print(points_xy[0][0], points_xy[0][1])
	    for i in range(len(points_xy)):
	    	self.X = np.append(self.X, points_xy[i][0])
	    	self.Y = np.append(self.Y, points_xy[i][1])
	    slope = m*self.X[1:] + b
	    plt.plot(self.X[1:], slope, '-r', label='Y='+str(m_trim)+'X+'+str(b_trim))#Y = mx+b
	    plt.scatter(self.X[1:], self.Y[1:], alpha=0.8)
	    plt.title('Plot of Regression Line')
	    plt.xlabel('sepal_width')
	    plt.ylabel('sepal_length')
	    plt.legend()
	    plt.show()

	def plot_gradient_Curve(self, grad_points, iteration, m_val):
	    #Display Gradient Curve with points
	    updated_m = np.round(m_val, 2)
	    update_grad_val = np.round(grad_points[1:], 1)
	    argmin = min(float(sub) for sub in update_grad_val)
	    max_step = np.where(update_grad_val == argmin)
	    max_step_value = max_step[0][0]
	    Random_Points = np.linspace(0,0.1,max_step_value)
	    #print(len(grad_points[1:max_step_value+1]),len(Random_Points))
	    plt.scatter(Random_Points, grad_points[1:max_step_value+1], alpha=0.8, label='Cost')
	    #print(Random_Points, update_grad_val, argmin, max_step, max_step_value, len(grad_points[1:max_step_value]),len(Random_Points)) #res_min = min(float(sub) for sub in test_list)
	    plt.title('Plot of Gradient Curve, Step = '+ str(max_step_value) ) #string_value = str(float_value)
	    plt.xlabel('Points')
	    plt.ylabel('Cost Function')
	    plt.legend()
	    plt.show()


	def compute_error(self, b, m, points):
	    #Calculate Errors in the start and at the end
	    #return average of all error points
	    totalError = 0
	    for i in range(0, len(points)):
	        x = points[i, 0]
	        y = points[i, 1]
	        totalError += (y - (m * x + b)) ** 2
	    return totalError / float(len(points))

	def step_gradient_calculate(self, b_current, m_current, points, learningRate):
	    #Calculate Gradient 
	    b_gradient = 0
	    m_gradient = 0
	    N = float(len(points))
	    for i in range(0, len(points)):
	        x = points[i, 0]
	        y = points[i, 1]
	        b_gradient += -(1/N) * (y - ((m_current * x) + b_current))
	        m_gradient += -(1/N) * x * (y - ((m_current * x) + b_current))
	    new_b = b_current - (learningRate * b_gradient)
	    new_m = m_current - (learningRate * m_gradient)
	    return [new_b, new_m]

	def gradient_descent_runner(self, points, starting_b, starting_m, LR_learning_rate, num_iterations):
	    #Calculate Steps for gradient 
	    b = starting_b
	    m = starting_m
	    gradient_plot = 0
	    for i in range(num_iterations):
	        b, m = self.step_gradient_calculate(b, m, points, LR_learning_rate)
	        gradient_parameter = self.compute_error(b, m, points)
	        gradient_plot = np.append(gradient_plot, gradient_parameter)
	    return [b, m, gradient_plot]

	def final_run(self, points):
	    #Initialize starting parameters, Display all parameters, Calculate Gradient for Line 
	    self.points = points
	    print("Initial Parameters: b = ", self.init_b,", m = ", self.init_m ,", error = ",self.compute_error(self.init_b, self.init_m, self.points), ", iterations = ", self.LR_Iteration)
	    print("After Calculation")
	    [b, m, self.gradient_plot_update] = self.gradient_descent_runner(self.points, self.init_b, self.init_m, self.LR_learning_rate, self.LR_Iteration)
	    print("b = ", b,", m = ", m, ", error = ",self.compute_error(b, m, points))
	    self.plot_graph(points, m, b)
	    self.plot_gradient_Curve(self.gradient_plot_update, self.LR_Iteration, m)

############################### Section-A Perceptron Model Basics #####################################################

class Perceptron:

	error = 0
	def __init__(self, learning_rate, iterations):
		#Initialize Instance variables and function
		self.lr = learning_rate
		self.iterations = iterations
		self.active = self.step_input
		self.weights = None
		self.bias = None
		self.error_Val = None

	def fit(self, X, y):
		#Fit method for training data (X) and respected output(y)
		#Training module for perceptron
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		y_ = np.array([1 if i > 0 else 0 for i in y])
		for _ in range(self.iterations):
			for idx, x_i in enumerate(X):
				lin_out = np.dot(x_i, self.weights)+self.bias
				y_predicted = self.active(lin_out)
				update = self.lr * (y_[idx] - y_predicted)
				self.error_Val =+ abs(lin_out*lin_out)
				self.weights += update * x_i
				self.bias += update
			self.error = np.append(self.error, int(self.error_Val)/n_samples)


	def predict(self, X):
		#Predict the resultant data with respect to training dataset
		lin_out = np.dot(X, self.weights) + self.bias
		y_predicted = self.active(lin_out)
		return y_predicted

	def step_input(self, x):
		#Convert data (x) into -1 and 1
		return np.where(x>=0, 1, -1)

	def net_input(self, X):
		#Display function for net_input
		lin_out = np.dot(X, self.weights) + self.bias
		print(lin_out)

	def plot_decision_regions(self, X, y, classifier, resolution=0.02):
		#Convert complete training data into points
		#plot points on 2d plane
		#Setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])
		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		
		# plot class samples
		for idx, cl in enumerate(np.unique(y)):
			plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
		#Display Result
		plt.xlabel('sepal length')
		plt.ylabel('petal length')
		plt.legend(loc='upper left')
		plt.show()

