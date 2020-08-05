import utils

class Person:
    def __init__(self, name, job, clearance):
        self.name = name
        self.job = job
        self.clearance = clearance

    def get_name(self):
        return self.name

    def get_job(self):
        return self.job

    def get_clearance(self):
        return self.clearance

    def get_creds(self):
        return [self.name, self.job, self.clearance]

class File:
    def __init__(self, path, file_type, confi_lvl):
        self.path = path
        self.file_type = file_type
        self.confi_lvl = confi_lvl

    def get_path(self):
        return self.path

    def get_file_type(self):
        return self.file_type

    def get_confi_lvl(self):
        return self.confi_lvl

    def get_meta_data(self):
        return [self.path, self.file_type, self.confi_lvl]

class Request:
    def __init__(self, date_time, geo, person, f):
        self.date_time = date_time
        self.geo = geo
        self.person = person
        self.f = f

    def get_request(self):
        nested_list = [
            self.date_time,
            self.geo,
            self.person.get_creds(),
            self.f.get_meta_data()]
        flat_list = list(utils.flatten(nested_list))
        return flat_list

class PolicyDecisionPoint:
    def __init__(self, policies):
        self.policies = policies

    def decision_response(self, req: Request) -> int:
        # access_granted = 1 if req.get_request()[2] in names[:10] else random.randint(0,1)
        # access_granted = 1 if req.get_request()[2] in names[:10] else 0
        # access_granted = random.randint(0,1)
        # return 1 if req.get_request()[2] in self.policies else random.randint(0,1)
        access = 0
        if self.policies['checkClearance']:
            if req.get_request()[5] >= req.get_request()[8]:
                access = 1
        elif req.get_request()[2] in self.policies['admins']:
            access = 1
        return access
