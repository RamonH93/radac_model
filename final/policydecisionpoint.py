from datetime import datetime
import pandas as pd

from generate_pips import (
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
            self.person_exists_policy,      # 1: 26984, 0: 3016
            # self.resource_exists_policy,    # 1: 30000, 0: 0
            self.location_policy,           # 1: 28502, 0: 1498
            self.is_owner_policy,           # 1: 21478, 0: 8522
            # self.age_policy,                # 1: 20961, 0: 9039
            # self.weekday_policy,            # 1: 21462, 0: 8538
            # self.time_policy,               # 1: 15005, 0: 14995
            self.clearance_policy,          # 1: 22700, 0: 7300
        ]

        self.active_policies = self.policies

    def evaluate(self, request):
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

    # 1: 21478, 0: 8522
    def is_owner_policy(self, request):
        return int(request['email'] == request['owner'])

    # TODO if not owner, is in own unit/dept
    def resource_in_unit(self, request):
        pass

    # 1: 20961, 0: 9039
    def age_policy(self, request):
        return int(request['age'] >= self.age_min and request['age'] <= self.age_max)

    # 1: 21462, 0: 8538
    def weekday_policy(self, request):
        return int(request['date'].weekday() in self.weekdays)

    # 1: 15005, 0: 14995
    def time_policy(self, request):
        return int(self.time_min <= request['time'].hour < self.time_max)

    # 1: 22700, 0: 7300
    def clearance_policy(self, request):
        return int(request['clearance_level'] >= request['confidentiality_level'])


def main():
    requests = pd.read_csv(FOLDER / 'requests.csv', index_col=0
                          ).astype({'date': 'datetime64', 'time': 'datetime64'})

    pdp = PolicyDecisionPoint()
    labels = []
    for i in requests.index:
        request = requests.iloc[i]
        id_ = str(request['id']).zfill(len(str(len(requests))))
        action = pdp.evaluate(request)
        labels.append({
            'id': id_,
            'action': action,
        })
        if i % (len(requests) / 10) == 0:
            print(f'{datetime.now()} Generated {i}/{len(requests)} actions.')

    print(pd.DataFrame(labels)['action'].value_counts())

    labels_df = pd.DataFrame(labels).set_index('id')
    labels_df.to_csv(FOLDER / 'labels.csv')
    print(f'{datetime.now()} Labels generated successfully.')


if __name__ == '__main__':
    main()
