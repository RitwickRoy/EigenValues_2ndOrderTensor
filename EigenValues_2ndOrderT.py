# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:40:47 2020

@author: Ritwick
"""
##
## Numpy arrays and built-functions
## Test Case:
## Compute eigen values of a symmetric 2nd order tensor
## 
import numpy as np
#
def FirstInvariant(a):
    #
    # First Invariant of Tensor a
    #
    I_1 = 0.0
    for i in range(n):
        I_1 += a[i][i]
    return I_1
#
def SecondInvariant(a):
    #
    # Second Invariant of Tensor a
    #
    I_2 = 0.0
    for i in range(n):
        for j in range(n):
            I_2 += a[i][i]*a[j][j]-a[i][j]*a[j][i]
    I_2 = 0.5*I_2
    return I_2
#
def UnitCyclicTensor():
    #   
    # Define a unit cyclic Tensor
    #
    uct = np.zeros((3,3,3))
    uct[0][1][2] = 1.0
    uct[1][2][0] = 1.0
    uct[2][0][1] = 1.0
    uct[2][1][0] = -1.0
    uct[0][2][1] = -1.0
    uct[1][0][2] = -1.0
    return uct
#
def ThirdInvariant(a):
    #
    # Third Invariant of Tensor a
    # This is also the determinant of matrix a
    #
    uct = UnitCyclicTensor()
    I_3 = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                I_3 += uct[i][j][k]*a[0][i]*a[1][j]*a[2][k]

    return I_3
#
def EigVal(I_1,I_2,I_3):
    #
    # The eigenvalues are the roots of the Cubic polynomial
    # e^3 - I_1*e^2 + I_2*e - I_3
    # The following steps are based on numerical recipes 
    # for computing roots of a cubic polynomial
    #
    one_third = 1.0/3.0
    pi = np.pi
    QQ = (I_1*I_1 - 3.0*I_2)/9.0
    RR = (-2.0*I_1*I_1*I_1 + 9.0*I_1*I_2 - 27.0*I_3)/54.0
    QQ3 = QQ*QQ*QQ
    RQ = RR / np.sqrt(QQ3)
    theta = np.arccos(RQ)
    two_sq = -2.0*np.sqrt(QQ)
    t0 = theta/3.0
    t1 = (theta+2*pi)/3.0
    t2 = (theta-2*pi)/3.0
    e1 = two_sq*np.cos(t0)  + one_third*I_1
    e2 = two_sq*np.cos(t1)  + one_third*I_1
    e3 = two_sq*np.cos(t2)  + one_third*I_1
#    EigenValues = -np.sort([-e1,-e2,-e3])
    e = [e1,e2,e3]
    EigenValues = np.sort(e)[::-1]
    return EigenValues
#
########################################################################
#
#    MAIN BODY of execution
#    STEPS:
#          1. generate a random 3X3 matrix
#          2. symmetrize the random matrix 
#          3. Compute invariants of the symmetrized matrix
#          4. Compute eigenvalues of the symmetric matrix
#          5. Call inbuild numpy function to compute eigenvalues for comparison
#          6. Ouput eigenvalues
#
########################################################################
#
#  generate a 3X3 random matrix b
#
n = 3
b= np.random.random((n,n))
#
#  symmetrize the matrix 
#  a = 1/2 *(b + b_transpose)
#
a = 0.5*(b + np.transpose(b))
#
# compute Invariants of symmetric matrix a
#
I_1 = FirstInvariant(a)
I_2 = SecondInvariant(a)
I_3 = ThirdInvariant(a)
#
# compute eigen values of a
#
e = []
e = EigVal(I_1,I_2,I_3)
#
#  Output Results
#
print('Symmetric Random Matrix:')
print()
print(a)
print()
print ('The 3 eigenvalues for the symmetric Matrix:')
print()
print('1st eigenvalue:',e[0])
print('2nd eigenvalue:',e[1])
print('3rd eigenvalue:',e[2])
print()
print ('Verify the 3 eigenvalues for the symmetric Matrix using Numpy:')
print()
e =np.linalg.eigvals(a)
print(e)
print()
print('det of Matrix:',I_3)
det =np.linalg.det(a)
print('Numpy det:',det)
