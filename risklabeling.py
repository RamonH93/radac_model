from functools import lru_cache
from itertools import combinations


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


# print(calc_sec_risk(tuple(p_normal)))

# from timeit import timeit

# print(timeit(lambda: is_even(2), number=1))
# print(timeit(lambda: is_even(2), number=1))

# # print(timeit(lambda: calc_sec_risk(tuple(p_elderly)), number=1))
# # print(timeit(lambda: calc_sec_risk(tuple(p_elderly)), number=1))
# print(calc_sec_risk.cache_info())
# print(or_node.cache_info())
# print(and_node.cache_info())










