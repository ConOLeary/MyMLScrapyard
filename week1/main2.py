import math
import csv
from numpy import *
import pygal

every_nvalues = 500

gradientd_graf = pygal.XY(stroke=False)
costvstime_graf = pygal.Line(x_title='Number of steps taken', y_title='Cost')
costvals = []

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
        if (i % every_nvalues == 0):
            linepoints = getLinePoints(b, m, points)
            gradientd_graf.add('Prediction', linepoints)
            costvals.append(compute_error_for_line_given_points(b, m, points))
    return [b, m]

####################################################################################################

def getLinePoints(b, m, vals):
    linepoints = []
    last_y = b
    linepoints.append((0, last_y))
    for i in range(len(vals) - 1):
        last_y = last_y + (m * 1)
        linepoints.append((i + 1, last_y))
    return linepoints

def run():
    y_vals = []
    xy_vals = []
    with open('week1.csv', 'r') as file:
        reader = csv.reader(file)
        next(file)
        i = 0
        for row in reader:
            y_vals.append(float(row[1]))
            xy_vals.append((i, float(row[1])))
            i = i + 1
    
    
    points = y_vals
    learning_rate = 0.000000002
    initial_b = 450 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    step_volume = 5000
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, step_volume)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(step_volume, b, m, compute_error_for_line_given_points(b, m, points)))

    initial_b = 450 # initial y-intercept guess
    initial_m = -0.4 # initial slope guess
    step_volume = 0
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, step_volume)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(step_volume, b, m, compute_error_for_line_given_points(b, m, points)))

    linepoints = getLinePoints(b, m, y_vals)

    xy_chart = pygal.XY(stroke=False)
    xy_chart.title = 'pls werk'
    xy_chart.add('Values', xy_vals)
    xy_chart.add('Prediction', linepoints)
    xy_chart.render_to_file('pls_werk.svg')

    gradientd_graf.title = 'gradient_descent_grafaroo'
    gradientd_graf.add('Values', xy_vals)
    gradientd_graf.render_to_file('gradientd_graf.svg')

    costvstime_graf.title = 'Cost function values for learning rate = 0.000000002'
    costvstime_graf.add('Cost values', costvals)
    costvstime_graf.x_labels = map(str, range(every_nvalues, (len(costvals) + 1) * every_nvalues)[0::every_nvalues])
    
    costvstime_graf.render_to_file('costvstime_graf.svg')

if __name__ == '__main__':
    run()