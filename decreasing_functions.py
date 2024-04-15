import numpy as np

def d50_revised_function(x, d50):
    return 48. + 48. / (1. + np.exp((x - d50) / (0.356 * d50) )) - 48. / (1. + np.exp(-1 / 0.356))

def d50_revised_calculate_parameter_function(x, y):
    return x / (1. + 0.356 * np.log(48 / (y - 48 +  48. / (1. + np.exp(-1 / 0.356)))- 1))

def rational_function(x, a):
    return 48. * a / (x + a)

def rational_calculate_parameter_function(x, y):
    return y * x / (48. - y)

def delta_function(x, delta):
    return 48. - delta * x

def delta_calculate_parameter_function(x, y):
    return (48. - y) / float(x)

def ln_function(x, a):
    return 48. - a * np.log(x + 1)

def ln_calculate_parameter_function(x, y):
    return (48. - y) / np.log(x + 1)

def calculate(function, x_data, parameters, parameters_2=np.array([])):
    function = globals()[function]
    if(parameters_2.any()):
        return function(x_data, parameters, parameters_2)
    return function(x_data, parameters)

def calculate_parameter(function, x_data, y_data):
    parameter_function = globals()[function]
    parameter = parameter_function(x_data, y_data)
    return parameter