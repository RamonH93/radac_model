import csv
import random
from pathlib import Path
import numpy as np
from faker import Faker
from classes import Person, File, Request, PolicyDecisionPoint


def generate_dataset(n=10000, seed=1, dest=Path("data.csv"), logger=None):
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    Path.mkdir(dest.parent, parents=True, exist_ok=False)

    fake = Faker(['nl_NL'])
    confi_lvls = clearance_lvls = np.arange(1, 6)
    confi_lvl_dist = clearance_dist = [0.05, 0.25, 0.2, 0.35, 0.15]

    # generate N/100 persons and files
    names = []
    files = []
    for _ in range(int(n/100)):
        name = fake.name()
        while name in names:
            name = fake.name()
        names.append(name)
        files.append(File(
            path=fake.file_path(random.randint(0, 2), 'office'),
            file_type=fake.mime_type('application'),
            confi_lvl=np.random.choice(confi_lvls, p=confi_lvl_dist)))

    persons = []
    for name in names:
        persons.append(Person(
            name=name,
            job=fake.job(),
            clearance=np.random.choice(clearance_lvls, p=clearance_dist)))

    pdp = PolicyDecisionPoint(policies={
        'admins': names[:10],
        'checkClearance': True})

    headers = [
        'ID',
        'date_time',
        'lat',
        'lng',
        'name',
        'job',
        'clearance',
        'path',
        'file_type',
        'confi_lvl',
        'access_granted'
        ]
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for id_ in range(n):
            req = Request(
                date_time=fake.date_time_between('-30d', 'now'),
                geo=fake.latlng(),
                person=persons[random.randint(0, len(persons)-1)],
                f=files[random.randint(0, len(files)-1)])

            access_granted = pdp.decision_response(req)

            row = []
            row.append(id_+1)
            row.extend(req.get_request())
            row.append(access_granted)
            writer.writerow(row)

    if logger:
        logger.info('Data generation successful')
