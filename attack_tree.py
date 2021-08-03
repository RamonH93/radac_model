from itertools import combinations
from functools import lru_cache
from scipy import interpolate
import random
import numpy as np

p_normal = [0.0024, 0.0024, 1.0E-6, 3.0E-5, 7.0E-5, 4.0E-5, 0.5, 0.001, 5.0E-5]
p_elderly = [0.024, 0.024, 0.001, 3.0E-5, 0.07, 0.04, 0.5, 0.01, 5.0E-5]


# @lru_cache()
def is_even(n):
    return n % 2 == 0


#   AND-NODE (Probability of n events)
#   1) Multiply the probabilities of the individual events.
@lru_cache()
def and_node(children):
    res = 1
    for child in children:
        res *= child
    return res


#   OR-NODE (Probability of union of n events):
#   1) Add the probabilities of the individual events.
#   2) Subtract the probabilities of the intersections of every pair of events.
#   3) Add the probabilities of the intersection of every set of three events.
#   4) Subtract the probabilities of the intersection of every set of four events.
#   5) Continue this process until the last probability is the probability of the intersection
#       of the total number of sets that we started with.
#   Source: https://www.thoughtco.com/probability-union-of-three-sets-more-3126263
@lru_cache()
def or_node(children):
    res = 0
    n = len(children)
    for i in range(1, n + 1):
        for comb in combinations(children, i):
            prod = and_node(comb)
            if is_even(i):
                res -= prod
            else:
                res += prod
    return res


# print(or_node([p_elderly[i] for i in [3, 4, 5, 6]]))


@lru_cache()
def calc_sec_risk(p):
    AB = or_node(p[1:3])
    A = and_node((p[0], AB))
    BA = or_node(p[3:7])
    BB = or_node(p[7:])
    B = and_node((BA, BB))
    R = or_node((A, B))
    return R


print(calc_sec_risk(tuple(p_normal)))

# from timeit import timeit

# print(timeit(lambda: is_even(2), number=1))
# print(timeit(lambda: is_even(2), number=1))

# # print(timeit(lambda: calc_sec_risk(tuple(p_elderly)), number=1))
# # print(timeit(lambda: calc_sec_risk(tuple(p_elderly)), number=1))
# print(calc_sec_risk.cache_info())
# print(or_node.cache_info())
# print(and_node.cache_info())


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

print({iage: p})

from bisect import bisect_right

class Interpolate:
    def __init__(self, x_list, y_list):
        if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
            raise ValueError("x_list must be in strictly ascending order!")
        self.x_list = x_list
        self.y_list = y_list
        intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
        self.slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    def __call__(self, x):
        if not self.x_list[0] <= x <= self.x_list[-1]:
            raise ValueError("x out of bounds!")
        if x == self.x_list[-1]:
            return self.y_list[-1]
        i = bisect_right(self.x_list, x) - 1
        return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])

interp = Interpolate(x, y)

pi = interp(iage)
print(pi)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlabel('Age')
plt.ylabel('$p_{Find note}=f(Age)$')
plt.show()

p_custom[1] = p

print(calc_sec_risk(tuple(p_custom)))
