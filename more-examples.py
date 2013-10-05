#!/usr/bin/python
import theano.tensor as T
from theano import function
import theano

## Logistic function
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x],s)
print logistic ([[0,1],[-1,-2]])

## Multiple outputs
a,b = T.dmatrices('a','b')
diff = a-b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a,b],[diff, abs_diff, diff_squared])
print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

## Default value for an argument
from theano import Param
x,y = T.dscalars('x','y')
z = x+y
sum = function([x,Param(y,default=1)],z)
print sum(33)
print sum(33,10)
# Similar to python inputs without default are first and then followed by ones with deafult vals. If multiple inputs with default values, then parameters are set by position or can be specified by the name

## Shared variables
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc],state,updates=[(state,state+inc)])
state.set_value(7)
print "After first accumuation",accumulator(5)
print accumulator(8)
print "State value",state.get_value()
print accumulator(0)


