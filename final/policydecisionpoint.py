from datetime import datetime
from functools import lru_cache
from itertools import combinations
import pandas as pd

from generate_pips import (
    CLEARANCE_LVLS,
    CONFIDENTIALITY_LVLS,
    DEFAULT_COMPANY,
    DEFAULT_COMPANY_LOCATION,
    FOLDER,
)
from policyinformationpoint import PolicyInformationPoint


class PolicyDecisionPoint:
    def __init__(self) -> None:
        self.pip = PolicyInformationPoint()
        self.weekdays = [0, 1, 2, 3, 4]
        self.company_location = DEFAULT_COMPANY_LOCATION[-2:]
        self.age_min = 18
        self.age_max = 85
        self.time_min = 7
        self.time_max = 19

        self.policies = [
            self.person_exists_policy,
            self.resource_exists_policy,
            self.location_policy,
            # self.is_owner_policy,
            self.resource_in_unit_policy,
            # self.resource_in_department_policy,
            self.owner_or_department_policy,
            # self.age_policy,
            self.weekday_policy,
            self.time_policy,
            self.company_policy,
            self.clearance_policy,
        ]

        self.active_policies = self.policies

        # person_exists_policy - 1: 28448, 0: 1552
        # resource_exists_policy - 1: 30000, 0: 0
        # location_policy - 1: 28459, 0: 1541
        #? is_owner_policy - 1: 22635, 0: 7365
        # resource_in_unit_policy - 1: 28149, 0: 1851
        #? resource_in_department_policy - 1: 21875, 0: 8125
        #* owner_or_department_policy - 1: 26564, 0: 3436
        #? age_policy - 1: 24148, 0: 5852
        # weekday_policy - 1: 28531, 0: 1469
        # time_policy - 1: 28593, 0: 1407
        # company_policy - 1: 28411, 0: 1589
        # clearance_policy - 1: 28568, 0: 1432

    def test_policies(self, requests, policy=None):
        to_test = self.active_policies if policy is None else [policy]
        for policy in to_test:
            labels = []
            for i in requests.index:
                request = requests.iloc[i-1]
                action = policy(request)
                labels.append(action)
            df = pd.Series(labels)
            try:
                n_permit = df.value_counts()[1]
            except KeyError:
                n_permit = 0
            try:
                n_deny = df.value_counts()[0]
            except KeyError:
                n_deny = 0
            print(f'{policy.__name__} - 1: {n_permit}, 0: {n_deny}')

    # 1: 21335, 0: 8665
    def evaluate(self, request):
        # C.1 Extended Indeterminate values
        # C.2 Deny-overrides
        # C.3 Ordered-deny-overrides
        # C.4 Permit-overrides
        # C.5 Ordered-permit-overrides
        # C.6 Deny-unless-permit
        #! C.7 Permit-unless-deny
        # C.8 First-applicable
        # C.9 Only-one-applicable
        for policy in self.active_policies:
            res = policy(request)
            if res == 0:
                return 0
        return 1

    # 1: 26984, 0: 3016
    def person_exists_policy(self, request):
        employee_data = self.pip.get_employee_attributes(request['email'])
        return int(employee_data is not None)

    # 1: 30000, 0: 0
    def resource_exists_policy(self, request):
        resource_data = self.pip.get_resource_attributes(request['resource'])
        return int(resource_data is not None)

    # 1: 28502, 0: 1498
    def location_policy(self, request):
        request_location = request['request_location']
        if request_location == request['country']:
            return 1
        elif request_location == self.company_location:
            return 1
        else:
            return 0

    def owner_or_department_policy(self, request):
        return (self.is_owner_policy(request) or self.resource_in_department_policy(request))

    # 1: 21478, 0: 8522
    def is_owner_policy(self, request):
        return int(request['email'] == request['owner'])

    # 1: 26923, 0: 3077
    def resource_in_unit_policy(self, request):
        return int(request['resource_unit'] == request['person_unit'])

    # 1: 20848, 0: 9152
    def resource_in_department_policy(self, request):
        return int(request['resource_department'] == request['person_department'])

    # 1: 20961, 0: 9039
    def age_policy(self, request):
        return int(request['age'] >= self.age_min)# and request['age'] <= self.age_max)

    # 1: 21462, 0: 8538
    def weekday_policy(self, request):
        return int(request['date'].weekday() in self.weekdays)

    # 1: 15005, 0: 14995
    def time_policy(self, request):
        return int(self.time_min <= request['time'].hour < self.time_max)

    # 1: 27111, 0: 2889
    def company_policy(self, request):
        return int(request['company'] == DEFAULT_COMPANY)

    # 1: 22700, 0: 7300
    def clearance_policy(self, request):
        return int(CLEARANCE_LVLS.index(request['clearance_level']) >= CONFIDENTIALITY_LVLS.index(request['confidentiality_level']))

class RiskAssessmentPoint:
    def __init__(self) -> None:
        self.pip = PolicyInformationPoint()
        self.p = [

        ]

    @lru_cache()
    def is_even(self, n):
        return n % 2 == 0

    #   AND-NODE (Probability of n events)
    #   1) Multiply the probabilities of the individual events.
    @lru_cache()
    def and_node(self, children):
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
    def or_node(self, children):
        res = 0
        n = len(children)
        for i in range(1, n + 1):
            for comb in combinations(children, i):
                prod = self.and_node(comb)
                if self.is_even(i):
                    res -= prod
                else:
                    res += prod
        return res

    def evaluate(self, request):
        import numpy as np
        return round(5 * np.random.rand(), 2)


def main():
    requests = pd.read_csv(FOLDER / 'requests.csv', index_col=0,
                          ).astype({'date': 'datetime64', 'time': 'datetime64'})
    pdp = PolicyDecisionPoint()
    rap = RiskAssessmentPoint()
    labels = []
    for i in requests.index:
        request = requests.iloc[i-1]
        id_ = str(i).zfill(len(str(len(requests))))
        action = pdp.evaluate(request)
        riskscore = rap.evaluate(request)
        labels.append({
            'id': id_,
            'action': action,
            'riskscore': riskscore,
        })
        if i % (len(requests) / 10) == 0:
            print(f'{datetime.now()} Generated {i}/{len(requests)} request labels.')

    print(f'1: {pd.DataFrame(labels)["action"].value_counts()[1]}, 0: {pd.DataFrame(labels)["action"].value_counts()[0]}')

    pd.DataFrame(labels).to_csv(FOLDER / 'labels.csv', index=False)
    print(f'{datetime.now()} Labels generated successfully.')


if __name__ == '__main__':
    main()
    # requests = pd.read_csv(FOLDER / 'requests.csv', index_col=0,
    #                       ).astype({'date': 'datetime64', 'time': 'datetime64'})
    # pdp = PolicyDecisionPoint()
    # pdp.test_policies(requests)
    # exists = []
    # owner = []
    # for i in requests.index:
    #     request = requests.iloc[i-1]
    #     exists.append(pdp.person_exists_policy(request))
    #     owner.append(pdp.is_owner_policy(request))
    # df = pd.DataFrame({'exists': exists, 'owner': owner})
    # print(df['exists'].value_counts())
    # print(df['owner'].value_counts())
    # print(df.loc[df['exists'] == df['owner'] & df['exists'] == 0])

