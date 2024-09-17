# import numpy
import numpy as np
# create a list
my_list = [24,8,3,4,34,8]
# convert the list to numpy array
np_list = np.array(my_list)
print(f"Original array before the np.unwrap(): {np_list}")
# unwrap each element and store it
np_list_unwrap = np.unwrap(np_list)
print("After the np.unwrap()")
print(np_list_unwrap)