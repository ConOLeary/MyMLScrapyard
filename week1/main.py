import math
import csv
from numpy import * # Related to data formatting
import pygal

def getLinePoints(b, m, vals):
    linepoints = []
    last_y = b
    linepoints.append((0, last_y))
    for i in range(len(vals) - 1):
        last_y = last_y + (m * 1)
        linepoints.append((i + 1, last_y))
    return linepoints

# y = mx + b
# m is slope, b is y-intercept
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

def gradient_descent_runner(vals, starting_b, starting_m, learning_rate, step_size):
    b = starting_b
    m = starting_m
    for i in range(step_size):
        b, m = step_gradient(b, m, vals, learning_rate)
    return [b, m]

def run(vals, learning_rate, step_size, initial_b, initial_m):
    print ("\n\nStarting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, vals)))
    print ("Running...")
    [b, m] = gradient_descent_runner(vals, initial_b, initial_m, learning_rate, step_size)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(step_size, b, m, compute_error_for_line_given_points(b, m, vals)))
    return [b, m]

if __name__ == '__main__':
    ################## Prepare data ##################
    vals = []
    with open('week1.csv', 'r') as file:
        reader = csv.reader(file)
        next(file)
        for row in reader:
            vals.append(round(float(row[1]), 5))
    '''
    for val in vals:
        print("val:",val)
    '''
    xy_array = []
    i = 0
    for val in vals:
        xy_array.append((i, val))
        i = i + 1

    # run(vals, learning_rate, step_size, initial_b, initial_m)
    ################## Graph 1 ##################
    [b, m] = run(vals, 1000000, 8, 450, -0.4)

    linepoints = getLinePoints(b, m, vals)
    xy_chart = pygal.XY(stroke=False)
    xy_chart.title = 'Learn rate = 0.001'
    xy_chart.add('Values', xy_array)
    xy_chart.add('Prediction', linepoints)
    xy_chart.render_to_file('smallLR.svg')

    ################## Graph 2 ##################
    [b, m] = run(vals, 1000000, 8, 450, -0.4)

    linepoints = getLinePoints(b, m, vals)
    xy_chart = pygal.XY(stroke=False)
    xy_chart.title = 'Learn rate = 0.01'
    xy_chart.add('Values', xy_array)
    xy_chart.add('Prediction', linepoints)
    xy_chart.render_to_file('mediumLR.svg')

    ################## Graph 3 ##################
    [b, m] = run(vals, 1000000, 0, 450, -0.4)

    linepoints = getLinePoints(b, m, vals)
    xy_chart = pygal.XY(stroke=False)
    xy_chart.title = 'Learn rate = 0.1'
    xy_chart.add('Values', xy_array)
    xy_chart.add('Prediction', linepoints)
    xy_chart.render_to_file('largeLR.svg')