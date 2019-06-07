# basic 1
''' -> input
9
1 2 3 4 5 6 7 8 9
10
pop
remove 9
discard 9
discard 8
remove 7
pop
discard 6
remove 5
pop
discard 5
'''
n = int(input())
s = set(map(int, input().split()))

for _ in range(int(input())):
    args = input().split()
    getattr(s, args[0])(*map(int, args[1:]))
print(s)

# basics 2
from itertools import groupby
print(*[(len(list(c)), int(k)) for k, c in groupby(input())])

# basic 3
import itertools
from itertools import combinations as c

n = int(input())
l = input().split()
k = int(input())

s = list(c(l, k))
f = filter(lambda d: 'a' in d, s)
print("{0:.3}".format(len(list(f))/len(s)))

# basic 4


from math import pow

n, mod = map(int, input().split())
top = []

for _ in range(n):
    l = input().split()
    top.append(sorted([int(x) for x in l], reverse=True))
    val = [d for d in top]
    t = [h[0] for h in val]
op = []
for k in t:
    op.append(pow(k, 2))
s = sum(op)
s = int(s) % mod
print(s)


'''
3 1000
2 5 4
3 7 8 9
5 5 7 8 9 10'''

# basic 5

for i in range(1,int(input())):
    print((10**(i)//9)*i)

# basic 6

n, m = map(int, input().split())
rows = [input() for _ in range(n)]
k = int(input())

for r in sorted(rows, key=lambda row: int(row.split()[k])):
    print(row)

# basic 7

# find palindrome

n = int(input())

arr = input().split()

print(all([int(i)>0 for i in arr])  and any([x == x[::-1] for x in arr]))

# baisc 7

# Sorting1234 -> ginortS1324
import string
print(*sorted(input(), key=(string.ascii_letters + '1357902468').index), sep='')


# basic 8

import numpy as np

n, m, p = map(int, input().split())

a = np.array([input().split() for _ in range(n)], int)
b = np.array([input().split() for _ in range(m)], int)

print(np.concatenate((a, b), axis = 0))

# basic 9

import numpy
print(str(numpy.eye(*map(int,input().split()))).replace('1',' 1').replace('0',' 0'))


# basic 10
# To print the pattern

n, m = map(int, input().split())

for i in range(1, n, 2):
    print((i*'.|.').center(m, '-'))
print(("WELCOME").center(m, '-'))
for i in range(n-2, -1, -2):
    print((i*'.|.').center(m, '-'))

'''

---------.|.---------
------.|..|..|.------
---.|..|..|..|..|.---
-------WELCOME-------
---.|..|..|..|..|.---
------.|..|..|.------
---------.|.---------

'''

# basic 9

import string

def print_rangoli(size):
    alpha = string.ascii_lowercase
    L = []
    for i in range(n):
        s = "-".join(alpha[i:n])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
    print('\n'.join(L[:0:-1]+L))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# basic 10

import numpy as np
n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')

# basic 11

import numpy as np
n, m = map(int, input().split())
arr = np.array([input().split(' ') for _ in range(0, n, m) for _ in range(m)]).astype(np.int)
val = np.sum(arr, axis = 0)
res = np.prod(val)
print(res)


# basic 12

from numpy import array
n, _m = map(int, input().split())
a = array([input().split() for _ in range(n)], int)
print(a.min(axis=1).max())

# basic 13

import numpy as np
from numpy import mean, var, std
np.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
arr = np.array([input().split() for _ in range(n)], int)
print(arr.mean(axis=1), arr.var(axis=0), arr.std(), sep = '\n')

# basic 14

import numpy as np

n = int(input())
a = np.array([input().split() for _ in range(n)], int)
b = np.array([input().split() for _ in range(n)], int)
print(np.dot(a, b))

# basic 15
import numpy

def arrays(arr):
    return numpy.flipud(numpy.array(arr, float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# basic 15
import numpy as np

arr = input().split(' ')
arr = list(map(int, arr)) # line convert str to int in a list

val = np.array(arr)
res = np.reshape(arr, (3,3))
print(res)

# basic 16
import numpy as np

n, m = map(int, input().split())
arr = np.array([input().split() for _ in range(m)], int)

print(np.transpose(arr))
print(arr.flatten())

# basic 17
print(__import__('numpy').polyval(list(map(float,input().split())),float(input())))

# basic 18

import numpy as np

import numpy
n=int(input())
a=numpy.array([input().split() for _ in range(n)],float)
numpy.set_printoptions(legacy='1.13') #important
print(numpy.linalg.det(a))


# basic 19

import re

pattern = re.compile('^[-+]?[0-9]*/.[0-9]+$')
for _ in range(int(input())):
    print(bool(pattern.match(input())))

# basic 20

import re
count = 0
decode = ''
val = list(map(int, input().split()))
input_val = [input() for x in range(val[0])]
while count != val[1]:
    for y in input_val:
        decode += y[count]
    count += 1
print(re.sub(r'(?<=([a-zA-Z0-9]))[\s$#%&]+(?=[a-zA-Z0-9])',' ', decode))


# basic 21

n, x = map(int, input().split())
sheet = []
for _ in range(x):
    sheet.append( map(float, input().split()) )
for i in zip(*sheet):
    print( sum(i)/len(i) )

# basic 22

for _ in range(int(input())):
x, a, z, b = input(), set(input().split()), input(), set(input().split())
print(a.issubset(b))

# basic 23

def fun(s):
    s_prcocess = s.replace('@','.').split('.')
    order = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1357902468-_'
    if len(s_prcocess) == 3 and all(s_items != '' for s_items in s_prcocess):
        if len(s_prcocess[2]) <= 3 and s_prcocess[1].isalnum() and all(item in order for item in s_prcocess[0]):
            return True
    return False

# basic 24

def product(fracs):
    t = reduce(lambda x, y : x * y, fracs)# complete this line with a reduce statement
    return t.numerator, t.denominator

# basic 25
return (f(spers) for spers in sorted((person[:2]+[int(person[2])]+person[3:]+[idx] for idx, person in enumerate(people)), key=operator.itemgetter(2, 4)))


# basic 26

import collections

numShoes = int(input())
shoes = collections.Counter(map(int, input().split()))
numCust = int(input())

income = 0

for i in range(numCust):
    size, price = map(int, input().split())
    if shoes[size]:
        income += price
        shoes[size] -= 1

print(income)

# basic 27

import re
for _ in range(int(input())):
    ans = True
    try:
        reg = re.compile(input())
    except re.error:
        ans = False
    print(ans)

# basic 28

for _ in range(int(input())):
    a, b = map(eval, input().split())
    try:
        val = a // b
        print(val)
    except BasicException as e:
        print('Error Code:',e)








