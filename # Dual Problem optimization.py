from scipy.optimize import linprog
import numpy as np
# define the function to minimize
# since its a linear problem, we define an array of the weights each variable has

wp = np.array([40, 30, 20, -30]) # so in our primal problem the function is min z = 40x1 + 30x2 + 20x3 -30x4

# define the constraints
# same thing with the constraints

Cu = np.array([[1, -4, 7, 0],
             [0, -6, 3, 0],
             [10, 4, -2, 6]])

Ce = np.array([0, 5, 3, 0])
# each of the lists represent the multipliers 
# for each of the constraints, which are unequalities (u) or equalities (e)

# we then define what the constraints should equal or bound to, and the variable bounds

Bu = np.array([10, 0, 40])
Be = np.array([4])

bounds = [(0, None), (0, None), (None, None), (-10**10, 10**10)]

# This is the representation of the primal. To include the dual problem we obtain it from these parameters.
# since the primal is a minimizing problem, the dual is a maximizing problem
# the variables for the dual come from the constraints of the primal:
# x1 -4x2 +7x3 +0x4 <= 10 --> y1 weights
# 0x1 -6x2 +3x3 +0x4 <= 0 --> y2 weights
# 10x1 +4x2 -2x3 +6x4 <= 40 --> y3 weights
# 0x1 +5x2 +3x3 +0x4 = 4 --> y4 weights

# The matrix of the constraints is then transposed to obtain the functions visually
'''
[[1, -4,  7, 0, 10],                [[10, 0,  4, 40],    which have       >=  40
 [0, -6,  3, 0,  0],     ----->      [0,  0,  6,  0],    the transposed   >=  30
 [10, 4, -2, 6,  4],                 [7, 3,  -2,  3],    primal function  >=  20
 [0,  5,  3, 0, 40]]                 [-4, -6, 4,  5]     as bounds        >= -30
                                     [1, 0,  10,  0]   
 
'''     

