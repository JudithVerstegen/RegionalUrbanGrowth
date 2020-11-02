import parameters
import numpy as np
import os

# Work directory
work_dir = parameters.getWorkDir()
    
# Define brute force calibration
def brute_force():
  # Prepeare empty list for the calibration parameters:
  parameters_list = []
  # Set step size for calibration
  min_p = parameters.getParametersforCalibration()[0]
  max_p = parameters.getParametersforCalibration()[1]
  stepsize = 0.05#parameters.getParametersforCalibration()[2]

  # Assure that steps in the loop have 3 decimal place only
  p_steps = np.around(np.arange(min_p, max_p+stepsize, stepsize),decimals=3)

  # Print calibration properties
  print('Case study: ', parameters.getCountryName())
  print('Number of iterations: ', parameters.getNumberofIterations())
  print('Min parameter value: ', min_p)
  print('Max parameter value: ', max_p)
  print('Parameter step: ', stepsize)
  # Get the possible combination of parameters:
  for p1,p2,p3,p4 in ((a,b,c,d) for a in p_steps for b in p_steps for c in p_steps for d in p_steps):
    sumOfParameters = p1+p2+p3+p4
    if (sumOfParameters > 0.9999 and sumOfParameters < 1.0001):
      parameters_list.append([p1,p2,p3,p4])

  # Return a list with parameters combinations
  return parameters_list


# Save the parameters as a txt file
text_file = os.path.join(work_dir,'parameters.txt')
params = brute_force()
with open(text_file, 'w') as f:
    for item in params:
        f.write("%s\n" % item)
f.close()
print(len(params))
