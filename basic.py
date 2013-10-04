#!/usr/bin/python
import theano.tensor as T
from theano import function
from theano import pp

# Working with scalars
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = function([x,y],z) # f is being compiled into C code
print f(2,3) 
print f(50,6)
print type(x)
print x.type
print x.type is T.dscalar
print pp(z) # print the computation associated to z
print type(f)

# Trying with matrices
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x,y],z) # f is being compiled into C code

print f([[1,2],[3,4]],[[10,20],[30,40]])

# Exercise
a = T.vector() # declare variable
out = a + a ** 10               # build symbolic expression
f = function([a], out)   # compile function
print f([0, 1, 2])  # prints `array([0, 2, 1026])`

a = T.vector()
b = T.vector()
out = a**2 + b**2 + 2*a*b
f = function([a,b],out)
print f([1,2],[4,5])


