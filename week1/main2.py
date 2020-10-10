import math
import csv
from numpy import *
import pygal
import sys

SHOW_EVERY_NSTEPS = 500

graf1 = pygal.XY(stroke=False)
graf2 = pygal.XY(stroke=False)
costvstime_graf = pygal.Line(x_title='Number of steps taken', y_title='Cost')
costvals = []

####################################################################################################
#################### Core Gradient Descent ####################
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = i
        y = points[i]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = i
        y = points[i]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, step_volume):
    b = starting_b
    m = starting_m
    for i in range(step_volume):
        b, m = step_gradient(b, m, points, learning_rate)
        if (i % SHOW_EVERY_NSTEPS == 0):
            linepoints = getLinePoints(b, m, points)
            graf1.add('Prediction', linepoints)
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

def addLineToGraf(b, m, amount_of_vals, graf):
    incrementby = 0.000001
    linepoints = []
    last_y = b
    linepoints.append((0, last_y))
    for i in range(amount_of_vals - 1):
        last_y = last_y + (m * 1) * incrementby
        linepoints.append(((i + 1)*incrementby, last_y))
    return linepoints

####################################################################################################
#################### Run ####################
def run():
    xy_vals = [[],[]]
    with open('week1.csv', 'r') as file:
        reader = csv.reader(file)
        next(file)
        i = 0
        for row in reader:
            xy_vals[0].append(i)
            xy_vals[1].append(float(row[1]))
            i = i + 1
    
    xy_vals[1] = normalise_data(xy_vals[1])
    '''for val in xy_vals[0]:
        print(val)'''
    xy_vals[0] = normalise_data(xy_vals[0])
    '''for val in xy_vals[0]:
        print(val)'''
    points = xy_vals[1]
    learning_rate = 0.000000002
    initial_b = 0.5 # initial y-intercept guess
    initial_m = -0.4 # initial slope guess
    amount_of_steps = 5000
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, amount_of_steps)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(amount_of_steps, b, m, compute_error_for_line_given_points(b, m, points)))
    
    xys = zip(xy_vals[0], xy_vals[1])
    graf1 = pygal.XY(stroke=False)
    graf1.title = 'pls werk'
    graf1.add('Values', xys)
    graf1.add('Prediction', linepoints)
    graf1.render_to_file('graf.svg')

    initial_b = 0.5 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    amount_of_steps = 0
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, amount_of_steps)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(amount_of_steps, b, m, compute_error_for_line_given_points(b, m, points)))

    linepoints = getLinePoints(b, m, xy_vals[1])
    '''for val in linepoints:
        print("val:",val)'''

    graf2.title = 'gradient_descent_grafaroo'
    graf2.add('Values', xys)
    graf2.render_to_file('graf2.svg')

    costvstime_graf.title = 'Cost function values for learning rate = 0.000000002'
    costvstime_graf.add('Cost values', costvals)
    costvstime_graf.x_labels = map(str, range(0, 1))
    
    costvstime_graf.render_to_file('costvstime_graf.svg')

if __name__ == '__main__':
    run()