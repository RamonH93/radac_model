from functools import lru_cache
import random
from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import risklabeling as rl

# 2020 age distribution in the Netherlands
# Respective groups: Under-5s, 5-14 years, 15-24 years, 25-64 years, 65+ years
# https://ourworldindata.org/age-structure
AGE_DIST = {
    0.0501: (0., 5.),
    0.1069: (5., 15.),
    0.1189: (15., 25.),
    0.5238: (25., 65.),
    0.2003: (65., 115 + (62 / 365))
}

p_normal = [0.0024, 0.0024, 1.0E-6, 3.0E-5, 7.0E-5, 4.0E-5, 0.5, 0.001, 5.0E-5]
p_elderly = [0.024, 0.024, 0.001, 3.0E-5, 0.07, 0.04, 0.5, 0.01, 5.0E-5]

def generate_age():
    age_group = np.random.choice(sorted(list(AGE_DIST.keys())),
                                 p=sorted(list(AGE_DIST.keys())))

    age = np.random.uniform(low=AGE_DIST[age_group][0],
                            high=AGE_DIST[age_group][1])

    return age


iage = generate_age()

p_custom = list(p_normal)

x = [v[1] for v in AGE_DIST.values()]
x.insert(0, AGE_DIST[min(AGE_DIST.keys())][0])
x = sorted(x)

y = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.024]

f = interpolate.interp1d(x, y)
p = float(f(iage))

print("age: scipy prob, " + str({iage: p}))
print("age: numpy prob, " + str({iage: np.interp(iage, x, y)}))

# already 20x faster than scipy's interp1d and 5x faster than np.interp
# lru_cache speeds it up even more
class Interpolate:
    def __init__(self, x_list, y_list):
        if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
            raise ValueError("x_list must be in strictly ascending order!")
        self.x_list = x_list
        self.y_list = y_list
        intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
        self.slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    @lru_cache()
    def __call__(self, x):
        if not self.x_list[0] <= x <= self.x_list[-1]:
            raise ValueError("x out of bounds!")
        if x == self.x_list[-1]:
            return self.y_list[-1]
        i = bisect_right(self.x_list, x) - 1
        return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])


interp = Interpolate(x, y)

pi = interp(iage)
print("age: class prob, " + str({iage: pi}))

# from timeit import timeit
# print(timeit(lambda: interpolate.interp1d(x, y)(iage), number=10000))
# print(timeit(lambda: f(iage), number=10000))
# print(timeit(lambda: np.interp(iage, x, y), number=10000))
# print(timeit(lambda: Interpolate(x, y)(iage), number=10000))
# print(timeit(lambda: interp(iage), number=10000))
# print(interp.__call__.cache_info())

# plt.plot(x, y)
# plt.xlabel('Age')
# plt.ylabel('$p_{Find note}=f(Age)$')
# plt.show()

p_custom[1] = p

print("secrisk=" + str(rl.calc_sec_risk(tuple(p_custom))))