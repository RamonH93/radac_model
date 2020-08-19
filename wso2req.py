import json
import requests
import urllib3
import zeep
import xmltodict

# import logging.config

# logging.config.dictConfig({
#     'version': 1,
#     'formatters': {
#         'verbose': {
#             'format': '%(name)s: %(message)s'
#         }
#     },
#     'handlers': {
#         'console': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#             'formatter': 'verbose',
#         },
#     },
#     'loggers': {
#         'zeep.transports': {
#             'level': 'DEBUG',
#             'propagate': True,
#             'handlers': ['console'],
#         },
#     }
# })

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# WSO2_ADMIN_API = 'https://localhost:9443/services/EntitlementAdminService?wsdl'
# WSO2_ADMIN_APII = 'https://localhost:9443/services/RemoteUserStoreManagerService?wsdl'
# WSO2_AUTH_API = "https://localhost:9443/api/identity/auth/v1.1/authenticate"
WSO2_PAP_SOAP_API = 'https://localhost:9443/services/EntitlementPolicyAdminService?wsdl'
WSO2_PDP_API = 'https://localhost:9443/api/identity/entitlement/decision/pdp'
WSO2_PDP_SOAP_API = 'https://localhost:9443/services/EntitlementService?wsdl'
USERNAME = "admin"
PASSWORD = "admin"
PROTOCOL = 'XML'
# PROTOCOL = 'JSON'

policy = """
<Policy xmlns="urn:oasis:names:tc:xacml:3.0:core:schema:wd-17" PolicyId="Patient-API-Policy" RuleCombiningAlgId="urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:first-applicable" Version="1.0">
    <Target>
        <AnyOf>
            <AllOf>
                <Match MatchId="urn:oasis:names:tc:xacml:1.0:function:string-regexp-match">
                    <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">/patient</AttributeValue>
                    <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:resource:resource-id" Category="urn:oasis:names:tc:xacml:3.0:attribute-category:resource" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                </Match>
            </AllOf>
        </AnyOf>
    </Target>
    <Rule Effect="Permit" RuleId="Rule-1">
        <Target>
            <AnyOf>
                <AllOf>
                    <Match MatchId="urn:oasis:names:tc:xacml:1.0:function:string-equal">
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">GET</AttributeValue>
                        <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:action:action-id" Category="urn:oasis:names:tc:xacml:3.0:attribute-category:action" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                    </Match>
                </AllOf>
            </AnyOf>
        </Target>
        <Condition>
            <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:any-of">
                <Function FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-equal"/>
                <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">Patient</AttributeValue>
                <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:subject:subject-id" Category="urn:oasis:names:tc:xacml:1.0:subject-category:access-subject" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
            </Apply>
        </Condition>
    </Rule>
    <Rule Effect="Permit" RuleId="Rule-2">
        <Condition>
            <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:and">
                <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-at-least-one-member-of">
                    <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-bag">
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">GET</AttributeValue>
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">POST</AttributeValue>
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">PUT</AttributeValue>
                    </Apply>
                    <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:action:action-id" Category="urn:oasis:names:tc:xacml:3.0:attribute-category:action" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                </Apply>
                <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:any-of">
                    <Function FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-equal"/>
                    <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">Physician</AttributeValue>
                    <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:subject:subject-id" Category="urn:oasis:names:tc:xacml:1.0:subject-category:access-subject" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                </Apply>
            </Apply>
        </Condition>
    </Rule>
    <Rule Effect="Permit" RuleId="Rule-3">
        <Condition>
            <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:and">
                <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-at-least-one-member-of">
                    <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-bag">
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">POST</AttributeValue>
                        <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">DELETE</AttributeValue>
                    </Apply>
                    <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:action:action-id" Category="urn:oasis:names:tc:xacml:3.0:attribute-category:action" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                </Apply>
                <Apply FunctionId="urn:oasis:names:tc:xacml:1.0:function:any-of">
                    <Function FunctionId="urn:oasis:names:tc:xacml:1.0:function:string-equal"/>
                    <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">Administrator</AttributeValue>
                    <AttributeDesignator AttributeId="urn:oasis:names:tc:xacml:1.0:subject:subject-id" Category="urn:oasis:names:tc:xacml:1.0:subject-category:access-subject" DataType="http://www.w3.org/2001/XMLSchema#string" MustBePresent="true"/>
                </Apply>
            </Apply>
        </Condition>
    </Rule>
    <Rule Effect="Deny" RuleId="Deny-Rule"/>
</Policy>
"""

request_json = {
    "Request": {
        "Resource": {
            "Attribute": [
                {
                    "AttributeId": "urn:oasis:names:tc:xacml:1.0:resource:resource-id",
                    "Value": "http://medi.com/patient/bob"
                }
            ]
        },
        "AccessSubject": {
            "Attribute": [
                {
                    "AttributeId": "urn:oasis:names:tc:xacml:1.0:subject:subject-id",
                    "Value": "Patient"
                }
            ]
        },
        "Action": {
            "Attribute": [
                {
                    "AttributeId": "urn:oasis:names:tc:xacml:1.0:action:action-id",
                    "Value": "GET"
                }
            ]
        }
    }
}

request_xml = """
<Request xmlns="urn:oasis:names:tc:xacml:3.0:core:schema:wd-17" CombinedDecision="false" ReturnPolicyIdList="false">
    <Attributes Category="urn:oasis:names:tc:xacml:3.0:attribute-category:resource">
        <Attribute AttributeId="urn:oasis:names:tc:xacml:1.0:resource:resource-id" IncludeInResult="false">
            <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">http://medi.com/patient/bob</AttributeValue>
        </Attribute>
    </Attributes>
    <Attributes Category="urn:oasis:names:tc:xacml:1.0:subject-category:access-subject">
        <Attribute AttributeId="urn:oasis:names:tc:xacml:1.0:subject:subject-id" IncludeInResult="false">
            <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">Patient</AttributeValue>
        </Attribute>
    </Attributes>
    <Attributes Category="urn:oasis:names:tc:xacml:3.0:attribute-category:action">
        <Attribute AttributeId="urn:oasis:names:tc:xacml:1.0:action:action-id" IncludeInResult="false">
            <AttributeValue DataType="http://www.w3.org/2001/XMLSchema#string">GET</AttributeValue>
        </Attribute>
    </Attributes>
</Request> 
"""


def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in
    this function because it is programmed to be pretty
    printed and may differ from the actual request.
    """
    print('{}\n{}\r\n{}\r\n\r\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))

# direct XML request, receive JSON response
print(requests.post(
    WSO2_PDP_API,
    headers={"Content-Type": "application/xml"},
    data=request_xml,
    auth=requests.auth.HTTPBasicAuth(USERNAME, PASSWORD),
    verify=False
    ).text)

# direct JSON request, receive JSON response
print(requests.post(
    WSO2_PDP_API,
    headers={"Content-Type": "application/json"},
    json=request_json,
    auth=requests.auth.HTTPBasicAuth(USERNAME, PASSWORD),
    verify=False
    ).text)

# prepared printable request, XML AND JSON, interpreted responses
r_json = requests.Request(
    'POST',
    WSO2_PDP_API,
    headers={"Content-Type": "application/json"},
    json=request_json,
    auth=requests.auth.HTTPBasicAuth(USERNAME, PASSWORD)
    )
r_xml = requests.Request(
    'POST',
    WSO2_PDP_API,
    headers={"Content-Type": "application/xml"},
    data=request_xml,
    auth=requests.auth.HTTPBasicAuth(USERNAME, PASSWORD)
    )
if PROTOCOL == 'JSON':
    prepped = r_json.prepare()
else:
    prepped = r_xml.prepare()
with requests.Session() as s:
    # pretty_print_POST(prepped)
    res = s.send(prepped, verify=False)
    if res.status_code == 200:
        if PROTOCOL == 'JSON':
            print(json.loads(res.text)['Response'][0]['Decision'])
        else:
            print(xmltodict.parse(res.text)['Response']['Result']['Decision'])
    else:
        print(res)

# SOAP communication with PAP: add, promote and remove policies
with requests.Session() as s:
    s.verify = False
    s.auth = requests.auth.HTTPBasicAuth(USERNAME, PASSWORD)
    try:
        transport = zeep.Transport(session=s)
        settings = zeep.Settings(raw_response=False)
        client = zeep.Client(WSO2_PAP_SOAP_API, transport=transport, settings=settings)
    except requests.exceptions.RequestException as e:
        print(e)

    factory = client.type_factory('ns1')
    policyDTO = factory.PolicyDTO()
    policyDTO.active = True
    policyDTO.policy = policy
    policyDTO.promote = True

    try:
        client.service.addPolicy(policyDTO)
        # client.service.removePolicy('Patient-API-Policy', dePromote=True)
    except zeep.exceptions.Error as e:
        print(e)

    print(client.service.getAllPolicyIds())
    # print(client.service.getPolicy('authn_scope_based_policy_template', False))


# SOAP communication with PDP: query the PDP
with requests.Session() as s:
    s.verify = False
    s.auth = requests.auth.HTTPBasicAuth(USERNAME, PASSWORD)
    try:
        transport = zeep.Transport(session=s)
        client = zeep.Client(WSO2_PDP_SOAP_API, transport=transport)
    except requests.exceptions.RequestException as e:
        print(e)

    try:
        res = client.service.getDecision(request=request_xml)
        print(xmltodict.parse(res)['Response']['Result']['Decision'])
        print(client.service.getBooleanDecision(
            subject='Patient',
            resource='/patient',
            action='DELETE'
            ))
        print(client.service.getDecisionByAttributes(
            subject='Patient',
            resource='/patient',
            action='DELETE'
            ))
        print(client.service.getEntitledAttributes(
            subjectName='Physician',
            # resourceName='/patient',
            # subjectId='urn:oasis:names:tc:xacml:1.0:subject:subject-id',
            # action='GET',
            enableChildSearch=False
        ))
    except zeep.exceptions.Error as e:
        print(e)
