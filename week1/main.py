import math
import csv
from numpy import *
import pygal
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

SHOW_EVERY_NSTEPS = 200

graf1 = pygal.XY(stroke=False)
graf2 = pygal.XY(stroke=False)
costvstime_graf = pygal.Line(x_title='Number of steps taken', y_title='Cost')
costvals = []

####################################################################################################
#################### Core Gradient Descent ####################
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(graf, points, starting_b, starting_m, learning_rate, amount_of_steps):
    b = starting_b
    m = starting_m
    addLineToGraf(graf, b, m, len(points))
    for i in range(amount_of_steps):
        [b, m] = step_gradient(b, m, points, learning_rate)
        #print("m is now: ",m)
        if (i % SHOW_EVERY_NSTEPS == 0):
            #print(">>>>>>inside SHOW_EVERY_NSTEPS loop")
            addLineToGraf(graf, (b), m, len(points))
            #print(xys)
            costvals.append(compute_error_for_line_given_points(b, m, points))
    return [b, m]

####################################################################################################
#################### Auxiliary Functions ####################
def normalise_data(vals):
    max = -sys.maxint - 1
    min = sys.maxint
    new_vals = []
    for val in vals:
        if val < min:
            min = val
        if val > max:
            max = val
    #mid = max - ((max - min) / 2) only need this line if we doing 1 -> -1
    for val in vals:
        val = float(val)
        val = (val - min) / (max - min)
        ''' only need these line if we doing 1 -> -1
        if val > 0:
            val = (val - mid) / ((max - min) * 0.5)
        if val < 0:
            val = (val - mid) / ((max - min) * 0.5)
        '''
        new_vals.append(val)
    return new_vals

def addLineToGraf(graf, b, m, amount_of_vals):
    #print(">>>inside addLineToGraf")
    amount_of_vals = int(amount_of_vals)
    incrementby = float(1) / amount_of_vals
    #incrementby = 1 # use this line instead on non-normalised data
    #print("amount_of_vals: ",amount_of_vals)
    #print("increment: ",float(1) / amount_of_vals)
    linepoints = []
    last_y = b
    linepoints.append((0, last_y))
    for i in range(amount_of_vals - 1):
        last_y = last_y + (m * incrementby)
        #print("point: (",(i + 1)*incrementby,",",last_y,")")
        linepoints.append(((i + 1)*incrementby, last_y))
    graf.add('Prediction', linepoints)
    #graf.add('TEST',[(0.4,0.4),(0.5,0.3),(0.6,0.2),(0.7,0.7),(0.8,0.8),(0.8,0.9)])
    return linepoints

####################################################################################################
#################### Run ####################
def run():
    xy_vals = [[],[]]
    norm_xy_vals = [[],[]]
    with open('week1.csv', 'r') as file:
        reader = csv.reader(file)
        next(file)
        i = 0
        for row in reader:
            xy_vals[0].append(i)
            xy_vals[1].append(float(row[1]))
            i = i + 1
    norm_xy_vals[1] = normalise_data(xy_vals[1])
    norm_xy_vals[0] = normalise_data(xy_vals[0])
    print("len(norm_xy_vals[1]): ",len(norm_xy_vals[1]))
    print("len(norm_xy_vals[0]): ",len(norm_xy_vals[0]))
    print("len(xy_vals[0]): ",len(xy_vals[0]))
    print("len(xy_vals[1]): ",len(xy_vals[1]))

    points = zip(norm_xy_vals[0], norm_xy_vals[1])
    graf1.add('Values', points)
    learning_rate = 0.0038
    initial_b = 0.5 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    amount_of_steps = 6000
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(graf1, points, initial_b, initial_m, learning_rate, amount_of_steps)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(amount_of_steps, b, m, compute_error_for_line_given_points(b, m, points)))
    graf1.title = 'pls werk'
    graf1.render_to_file('graf1.svg')
    
    initial_b = 0.5 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    amount_of_steps = 0
    graf2.add('Values', points)
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(graf2, points, initial_b, initial_m, learning_rate, amount_of_steps)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(amount_of_steps, b, m, compute_error_for_line_given_points(b, m, points)))
    graf2.title = 'gradient_descent_grafaroo'
    graf2.render_to_file('graf2.svg')
    
    Xs = pd.DataFrame(norm_xy_vals[0], xy_vals[0])#.T
    Ys = pd.DataFrame(norm_xy_vals[1], xy_vals[0])
    print("\n Ys")
    print(Ys)
    print("\n Xs")
    print(Xs)
    
    plt.interactive(True)
    model = LinearRegression()
    model.fit(Xs, Ys)
    plt.scatter(Xs, Ys)
    plt.plot(Xs, model.predict(Xs))
    plt.show()

    costvstime_graf.title = 'Cost function values for learning rate = 0.0038'
    costvstime_graf.add('Cost values', costvals)
    costvstime_graf.x_labels = map(str, range(0, int(amount_of_steps/SHOW_EVERY_NSTEPS)))
    costvstime_graf.render_to_file('costvstime_graf.svg')

if __name__ == '__main__':
    run()