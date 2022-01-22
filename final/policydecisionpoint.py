import pandas as pd

from generate_pips import (
    DEFAULT_COMPANY_LOCATION,
    FOLDER,
)
from policyinformationpoint import PolicyInformationPoint

class PolicyDecisionPoint:
    def __init__(self) -> None:
        self.pip = PolicyInformationPoint()

        self.policies = [
            self.person_exists_policy,
            self.resource_exists_policy,
            self.location_policy,
            self.is_owner_policy,
        ]

        self.active_policies = self.policies

    def evaluate(self, request):
        for policy in self.active_policies:
            res = policy(request)
            if res == 0:
                return 0
        return 1

    def person_exists_policy(self, request):
        email = request['email']
        employee_data = self.pip.get_employee_attributes(email)
        return int(employee_data is not None)

    #TODO age policy

    #TODO time policy

    #TODO clearance level policy

    def location_policy(self, request):
        native_country = request['country']
        request_location = request['request_location']
        if request_location == native_country:
            return 1
        elif request_location == DEFAULT_COMPANY_LOCATION[-2:]:
            return 1
        else:
            return 0

    def resource_exists_policy(self, request):
        resource = request['resource']
        resource_data = self.pip.get_resource_attributes(resource).to_dict(orient='records')[0]
        return int(resource_data is not None)

    def is_owner_policy(self, request):
        email = request['email']
        resource = request['resource']
        resource_data = self.pip.get_resource_attributes(resource).to_dict(orient='records')[0]
        owner = resource_data['owner']
        return int(email == owner)

def main():
    requests = pd.read_csv(FOLDER / 'requests.csv', index_col=0).to_dict(orient='records')
    # employee = 'aronpost@tno.nl'
    # employee_data = pip.get_employee_attributes(employee).to_dict(orient='records')[0]
    # resource = 'either.mp3'
    # resource_data = pip.get_resource_attributes(resource).to_dict(orient='records')[0]
    # print(employee_data)
    # print(employee_data['person'])
    # print(resource_data)

    pdp = PolicyDecisionPoint()
    labels = []
    for i, request in enumerate(requests):
        id_ = str(request['id']).zfill(len(str(len(requests))))
        action = pdp.evaluate(request)
        labels.append({
            'id': id_,
            'action': action,
        })
        if i % (len(requests) / 10) == 0:
            print(f'Generated {i}/{len(requests)} actions.')

    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(FOLDER / 'labels.csv')
    print('Labels generated successfully.')

    labels = pd.read_csv(FOLDER / 'labels.csv', index_col=0)
    print(labels['action'].value_counts())

if __name__ == '__main__':
    main()
