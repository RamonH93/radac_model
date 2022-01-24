import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider

# from faker.exceptions import UniquenessException

### SETTINGS
SEED = 0
NUM_EMPLOYEES = 2000
NUM_RESOURCES = 10000
NUM_REQUESTS = 30000
FOLDER = Path('final')

# EMPLOYEES PROBABILITIES
P_NATIONALITY = [0.5, 0.5] # of having same nationality as default company
P_COMPANY = [0.95, 0.05] # of working for the default company or another
P_CLEARANCE_LVLS = [0.3, 0.25, 0.2, 0.15, 0.1] # unclassified to top secret

# RESOURCES PROBABILITIES
P_FILE_OWN_DEPT = 0.95 # p file created within own department or other within unit
P_CONFIDENTIALITY_LVLS = P_CLEARANCE_LVLS  # unclassified to top secret

# REQUESTS PROBABILITIES
P_BUSINESS_DAYS = [0.95, 0.05]
P_WORKING_HOURS = [0.95, 0.05]
P_INSIDER = [0.95, 0.05]
P_INSIDER_RESOURCE = [0.8, 0.15, 0.04, 0.01] # own/department/unit/other resource
P_EMPLOYEE_LOCATION = [0.7, 0.25, 0.05] # own/company/other country

### IMMUTABLES
# Only locales that initialize properly with the faker module
LOCALES = [
    'ar_AE',
    'ar_EG',
    'ar_JO',
    # 'ar_PS',
    # 'ar_SA',
    # 'az_AZ',
    # 'bg_BG',
    'bn_BD',
    'bs_BA',
    # 'cs_CZ', # WEKA
    'da_DK',
    'de_AT',
    'de_CH',
    'de_DE',
    'el_CY',
    # 'el_GR',
    'en_AU',
    'en_CA',
    'en_GB',
    'en_IE',
    'en_IN',
    'en_NZ',
    'en_PH',
    'en_US',
    'es_CO',
    # 'es_ES', # WEKA
    'es_MX',
    'et_EE',
    # 'fa_IR',
    'fi_FI',
    'fil_PH',
    'fr_CA',
    'fr_CH',
    'fr_FR',
    'ga_IE',
    # 'he_IL',
    # 'hi_IN',
    # 'hr_HR', # WEKA
    # 'hu_HU', # WEKA
    # 'hy_AM',
    'id_ID',
    'it_CH',
    'it_IT',
    # 'iw_IL',
    # 'ja_JP',
    # 'ka_GE',
    # 'ko_KR',
    'lb_LU',
    'lt_LT',
    # 'lv_LV', # WEKA
    'mt_MT',
    # 'ne_NP',
    'nl_BE',
    'nl_NL',
    # 'or_IN',
    'pl_PL',
    'pt_BR',
    'pt_PT',
    'ro_RO',
    # 'ru_RU',
    'sk_SK',
    # 'sl_SI', # WEKA
    'sv_SE',
    # 'ta_IN',
    # 'th_TH',
    'tl_PH',
    'tr_TR',
    # 'uk_UA',
    # 'zh_CN',
    # 'zh_TW',
]

DEFAULT_COMPANY = 'TNO'
DEFAULT_COMPANY_LOCATION = 'nl_NL'
CLEARANCE_LVLS = ['None', 'VOG', 'VGB-C', 'VGB-B', 'VGB-A']
CONFIDENTIALITY_LVLS = ['Unclassified', 'Restricted', 'Confidential', 'Secret', 'Top Secret']
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
JOB_PREFIX = ['Junior', 'Medior', 'Senior',]
JOB_TITLES = [
    'Researcher',
    'Manager',
    'Consultant',
    'Scientist',
]
# https://www.tno.nl/en/about-tno/organisation/
UNITS = [
    'Artificial Intelligence',
    'Defence, Safety & Security',
    'Industry',
    'Healthy Living',
    'Traffic & Transport',
    'Information & Communication Technology',
]
DEPARTMENTS = {
    'Artificial Intelligence': [
        'Personalised Health',
        'Autonomous Vehicles and Systems',
        'Cyber Security',
        'Predictive Maintenance',
    ],
    'Defence, Safety & Security': [
        'Acoustics and Sonar',
        'Electronic Defence',
        'Intelligent Imaging',
        'Military Operations',
    ],
    'Industry': [
        'Space Systems Engineering',
        'Nano Instrumentation',
        'Quantum Technology',
        'Food and Pharma Printing',
    ],
    'Healthy Living': [
        'Microbiology & Systems Biology',
        'Child Health',
        'Work Health Technology',
        'Sustainable Productivity and Employabillity',
    ],
    'Traffic & Transport': [
        'Research on Integrated Vehicle Safety',
        'Sustainable Transport and Logistics',
        'Sustainable Urban Mobility and Safety',
    ],
    'Information & Communication Technology': [
        'Data Science',
        'Networks',
        'Monitoring & Control Services',
        'Cyber Security & Robustness',
        'Internet of Things',
    ],
}
RESOURCE_CATEGORIES = ['audio', 'image', 'office', 'text', 'video']

### FUNCTIONS
def email_format(s: str) -> str:
    return "".join(s.split()).lower().translate({ord(c): "" for c in "&,."})


### CLASSES
class MyProvider(BaseProvider):

    def age(self) -> int:
        age_group = np.random.choice(sorted(list(AGE_DIST.keys())),
                                     p=sorted(list(AGE_DIST.keys())))
        return np.random.randint(low=AGE_DIST[age_group][0],
                                 high=AGE_DIST[age_group][1])

    def department(self, unit: str) -> str:
        return np.random.choice(DEPARTMENTS[unit])

    def job(self) -> str:
        return np.random.choice(JOB_PREFIX) + ' ' + np.random.choice(JOB_TITLES)

    def date(self) -> str:
        start = datetime(2022, 1, 1)
        end = datetime(2022, 2, 1)
        businessdays = pd.bdate_range(start=start, end=end)
        alldays = pd.date_range(start=start, end=end)
        weekenddays = np.delete(alldays, np.argwhere(np.isin(alldays, businessdays)))
        day = np.random.choice(np.random.choice(np.array([businessdays, weekenddays], dtype=object), p=P_BUSINESS_DAYS))
        return pd.to_datetime(str(day)).strftime('%Y-%m-%d')

    def time(self) -> str:
        working_hours = range(8, 19)
        other_hours = np.append(range(8), (range(19, 24)))
        hour = np.random.choice(np.random.choice(np.array([working_hours, other_hours], dtype=object), p=P_WORKING_HOURS))
        minute = np.random.randint(60)
        second = np.random.randint(60)
        return datetime.strptime(f'{hour}:{minute}:{second}', '%H:%M:%S')

    def unit(self) -> str:
        return np.random.choice(UNITS)


def generate_employee(fake: Faker=None) -> dict:
    if fake is None:
        locale = np.random.choice([DEFAULT_COMPANY_LOCATION, np.random.choice(LOCALES)], p=P_NATIONALITY)
        fake = Faker(locale)
        fake.add_provider(MyProvider)
    name = fake.unique.name()
    age = fake.age()
    country = fake.current_country_code()
    city = fake.city()
    company = np.random.choice([DEFAULT_COMPANY, fake.company()], p=P_COMPANY)
    unit_ = fake.unit()
    department = fake.department(unit_)
    job = fake.job()
    email = (f'{email_format(name)}@{email_format(company)}'
             f'.{"nl" if company == "TNO" else fake.tld()}')
    try:
        phone = fake.phone_number()
    except AttributeError:
        phone = fake.numerify('##########')
    clearance_lvl = np.random.choice(CLEARANCE_LVLS, p=P_CLEARANCE_LVLS)

    employee = {
        'email': email,
        'name': name,
        'age': age,
        'country': country,
        'city': city,
        'company': company,
        'person_unit': unit_,
        'person_department': department,
        'job': job,
        'phone': phone,
        'clearance_level': clearance_lvl,
    }
    return employee

### EMPLOYEES GENERATION
def generate_employees():
    fakers = {}
    # for locale in LOCALES:
    #     faker[locale] = Faker(locale)
    employees = []

    for i in range(NUM_EMPLOYEES):
        locale = np.random.choice([DEFAULT_COMPANY_LOCATION, np.random.choice(LOCALES)], p=P_NATIONALITY)
        if locale not in fakers.keys():
            fakers[locale] = Faker(locale)
            fakers[locale].add_provider(MyProvider)
        fake = fakers[locale]
        employee = generate_employee(fake)

        employees.append(employee)

        if i % (NUM_EMPLOYEES / 10) == 0:
            print(f'{datetime.now()} Generated {i}/{NUM_EMPLOYEES} employees.')

    employees_df = pd.DataFrame(employees)
    return employees_df


### RESOURCES GENERATION
def generate_resources(employees_df=None):
    fake = Faker()
    if employees_df is None:
        employees_df = pd.read_csv(FOLDER / 'employees.csv', index_col=0)
    resources = []

    for i in range(NUM_RESOURCES):
        category = np.random.choice(RESOURCE_CATEGORIES)
        # Resources are unique
        resource = fake.unique.file_name(category=category)
        owner = np.random.choice(employees_df['email'].values)
        # Owners make files for the same unit
        unit__ = employees_df.loc[employees_df['email'] == owner, 'person_unit'].values[0]
        # Department can be different through cooperation
        p_departments = []
        own_department = employees_df.loc[employees_df['email'] == owner, 'person_department'].values[0]
        own_dept_idx = DEPARTMENTS[unit__].index(own_department)
        num_depts = len(DEPARTMENTS[unit__])
        p_other_depts = (1 - P_FILE_OWN_DEPT) / (num_depts - 1)
        for j in range(num_depts):
            if j == own_dept_idx:
                p_departments.append(P_FILE_OWN_DEPT)
            else:
                p_departments.append(p_other_depts)
        department_ = np.random.choice(DEPARTMENTS[unit__], p=p_departments)
        owner_clearance = employees_df.loc[employees_df['email'] == owner]['clearance_level'].values[0]
        confi_start_idx = CLEARANCE_LVLS.index(owner_clearance)
        confidentiality_lvl = np.random.choice(CONFIDENTIALITY_LVLS[:confi_start_idx+1])

        resources.append({
            'resource': resource,
            'owner': owner,
            'category': category,
            'resource_unit': unit__,
            'resource_department': department_,
            'confidentiality_level': confidentiality_lvl,
        })

        if i % (NUM_RESOURCES / 10) == 0:
            print(f'{datetime.now()} Generated {i}/{NUM_RESOURCES} resources.')

    resources_df = pd.DataFrame(resources)
    return resources_df


def generate_requests(employees_df=None, resources_df=None):
    fakers ={}
    locale = np.random.choice(LOCALES)
    fakers[locale] = Faker(locale)
    fakers[locale].add_provider(MyProvider)
    fake = fakers[locale]
    if employees_df is None:
        employees_df = pd.read_csv(FOLDER / 'employees.csv', index_col=0)
    if resources_df is None:
        resources_df = pd.read_csv(FOLDER / 'resources.csv', index_col=0)
    requests = []

    for i in range(1, NUM_REQUESTS + 1):
        # for i in range(1, 100):

        ### generate id, date and time
        id_ = str(i).zfill(len(str(NUM_REQUESTS)))
        date = fake.date()
        time = fake.time()

        #high chance of insider vs outsider
        insider = employees_df.sample().to_dict(orient='records')[0]
        outsider = None
        person = np.random.choice(np.array([insider, outsider], dtype=object), p=P_INSIDER)
        if person is not None:
            ### Assign resource to insider, p=INSIDER_RESOURCE
            own_resources = resources_df.loc[resources_df['owner'] == person['email']]['resource'].values
            department = person['person_department']
            department_resources = resources_df.loc[resources_df['resource_department'] == department]['resource'].values
            if len(own_resources) > 0:
                department_resources_excl_own = np.delete(department_resources, np.argwhere(np.isin(department_resources, own_resources)))
                # department_resources_excl_own = resources_df.loc[(resources_df['department'] == department) & ~resources_df['resource'].isin(own_resources)]['resource'].values # way slower (0.08 vs 1.87 for 1000 iterations)
            else:
                department_resources_excl_own = department_resources
            unit = person['person_unit']
            # unit_resources = resources_df.loc[resources_df['unit'] == unit]['resource'].values
            # unit_resources_excl_own_excl_dept = np.delete(unit_resources, np.append(np.argwhere(np.isin(unit_resources, own_resources)), np.argwhere(np.isin(unit_resources, department_resources_excl_own)))) # way slower (22.3 vs 3.77 for 1000 iterations)
            unit_resources_excl_own_excl_dept = resources_df.loc[(resources_df['resource_unit'] == unit) & (resources_df['resource_department'] != department) & (~resources_df['resource'].isin(own_resources))]['resource'].values
            other_resources = resources_df.loc[(resources_df['resource_unit'] != unit) & ~resources_df['resource'].isin(own_resources)]['resource'].values
            # other_resources = np.delete(resources_df['resource'].values, np.argwhere(np.isin(resources_df['resource'].values, unit_resources))) # way slower (1.69 for 1000 iterations vs 3.42 for 10 iterations)
            try:
                own_resource = np.random.choice(own_resources)
            except ValueError:
                own_resource = np.random.choice(department_resources)
            department_resource = np.random.choice(department_resources_excl_own)
            unit_resource = np.random.choice(unit_resources_excl_own_excl_dept)
            other_resource = np.random.choice(other_resources)
            resource = np.random.choice([own_resource, department_resource, unit_resource, other_resource], p=P_INSIDER_RESOURCE)
        else:
            ### generate outsider, request random resource
            locale = np.random.choice(LOCALES)
            if locale not in fakers.keys():
                fakers[locale] = Faker(locale)
                fakers[locale].add_provider(MyProvider)
            fake = fakers[locale]
            person = generate_employee(fake)
            while person['email'] in employees_df['email']:
                person = generate_employee(fake)
            resource = np.random.choice(resources_df['resource'])

        ### generate request location
        native_country = person['country']
        other_location = np.random.choice(LOCALES)[-2:]
        while other_location == native_country or other_location == DEFAULT_COMPANY_LOCATION[-2:]:
            other_location = np.random.choice(LOCALES)[-2:]
        request_location = np.random.choice([
            native_country,
            DEFAULT_COMPANY_LOCATION[-2:],
            other_location
            ], p=P_EMPLOYEE_LOCATION)

        ### construct request
        request = {
            'id': id_,
            'date': date,
            'time': time,
            'request_location': request_location,
        }
        resource_data = resources_df.loc[resources_df['resource'] == resource].to_dict(orient='records')[0]
        for k, v in resource_data.items():
            request[k] = v
        for k, v in person.items():
            request[k] = v
        requests.append(request)

        if i % (NUM_REQUESTS / 10) == 0:
            print(f'{datetime.now()} Generated {i}/{NUM_REQUESTS} requests.')

    requests_df = pd.DataFrame(requests)
    return requests_df

def main():
    ### INIT
    if SEED is not None:
        Faker.seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    employees_df_ = generate_employees()
    while employees_df_['email'].nunique() != employees_df_['email'].count():
        print(f'{datetime.now()} Email addresses not unique, trying again.')
        employees_df_ = generate_employees()
    employees_df_.to_csv(FOLDER / 'employees.csv')
    print(f'{datetime.now()} Employees generated successfully.')

    resources_df_ = generate_resources(employees_df_)
    while resources_df_['resource'].nunique() != resources_df_['resource'].count():
        print(f'{datetime.now()} Resources not unique, trying again.')
        resources_df_ = generate_resources(employees_df_)
    resources_df_.to_csv(FOLDER / 'resources.csv')
    print(f'{datetime.now()} Resources generated successfully.')

    requests_df = generate_requests()
    requests_df.to_csv(FOLDER / 'requests.csv', index=False)
    print(f'{datetime.now()} Requests generated successfully.')


if __name__ == '__main__':
    main()
