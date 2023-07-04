---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: doctor
    language: python
    name: python3
---

```python
# https://docs.python.org/3/library/fractions.html
from fractions import Fraction
# https://docs.python.org/3/library/decimal.html
from decimal import Decimal
import decimal
from quasiPolynomial import QuasiPolynomial, Polynomial
```

```python
# Jupyter notebook for issue #68
```

```python
# https://stackoverflow.com/questions/7559595/python-runtimewarning-overflow-encountered-in-long-scalars 
# The starting case. One *may* obtain an error when performing an overflowing operation with the numpy types
import numpy as np
np.seterr(all='warn')
B = np.array([144],dtype='double')
b=B[-1]
print ("Hello python:", b**b)
```

```python
# For Decimal an overflow is not possible in principle
decimal.getcontext().prec=50
Decimal(B[0])**2000
```

```python
fr = Fraction(1/B[-1])
type(fr.denominator)
```

```python
# The `Decimal` type is converted to `int`
f = Fraction(Decimal(2))
type(f.numerator)
```

```python
# Integer objects are implemented as "long" integer objects of arbitrary size. So, they should not overflow.
# https://docs.python.org/3/c-api/long.html#integer-objects
144**144
```

```python
# If we use the int64 type of numpy, we again obtain an overflow.
B = np.array([144],dtype='int')
print(type(B[-1]))
B[-1]**B[-1]
```

```python
# If we put the int64 into the Fraction it is *not* converted and we obtain again an overlow
B = np.array([144],dtype='int')
f = Fraction(B[-1])
print(type(f.numerator))
print(f**f)
# But we can convert the type beforehand so that the result stays correct
g = Fraction(int(B[-1]))
print(g**g)
```

```python
# Using a check `isinstance(.,int)` one can check if a 'proper' int type is used.
isinstance(B[-1],np.int64), isinstance(B[-1],int)
```

```python
# Type of Fraction in Polynomial remains the same
# Site note: I don't understand why the int16 is expressed as `<class 'numpy.int64'>`
fr = Fraction(*np.array([1,4],dtype=np.int16))
print(type(fr.denominator))
poly = Polynomial.new(coefficient_list=[2,fr])
print(poly.pretty_print())
print(type(poly.coefficients()[1].denominator))
```

```python
# Fix above problem by recreating the Fraction
fr = Fraction(*np.array([1,4],dtype=np.int16))
print(type(fr.denominator))
fr = Fraction(int(fr.numerator), int(fr.denominator))
poly = Polynomial.new(coefficient_list=[2,fr])
print(poly.pretty_print())
print(type(poly.coefficients()[1].denominator))
```

```python
# Comparison of two versions, once with `np.int64` and once with `int`
# First result is wrong an throws a RuntimeWarning for overflow
fr = Fraction(*np.array([1,1000000],dtype=np.int64))
poly = Polynomial.new(coefficient_list=[fr])
poly = poly*poly*poly*poly
print(poly.pretty_print())
fr = Fraction(1,1000000)
poly = Polynomial.new(coefficient_list=[fr])
poly = poly*poly*poly*poly
print(poly.pretty_print())
```

```python
print(type(np.asarray([1,23])[0]))
print(type(np.asarray([1,23],dtype=int)[0]))
print(type(np.asarray([1,23],dtype=float)[0]))
print(type(np.asarray([1,23],dtype=int).tolist()[0])) #This seems to work
print(type(np.asarray([1,23],dtype=np.int64).tolist()[0])) #This seems to work
print(type(np.asarray([1,23],dtype=int).astype(int)[0]))
```

```python
np.asarray([]).astype(Fraction)
```

```python
list(np.asarray([1,23]))
```

```python
fr = Fraction(1,1000000)
poly = Polynomial.new(coefficient_list=[fr])
type((poly * np.asarray([20])[0]).coefficients()[0].denominator)
```
