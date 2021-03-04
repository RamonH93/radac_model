from cryptography.fernet import Fernet
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


def send_notification(first, second):
    try:
        key = open("twilio.key", "rb").read()
        f = Fernet(key)
        with open("twilio.token", "rb") as file:
            account_sid_encrypted,\
                 auth_token_encrypted, from_encrypted, to_encrypted = file.readlines()

        account_sid = f.decrypt(account_sid_encrypted).decode()
        auth_token = f.decrypt(auth_token_encrypted).decode()
        from_ = f.decrypt(from_encrypted).decode()
        to = f.decrypt(to_encrypted).decode()

        client = Client(account_sid, auth_token)

        message = client.messages.create(
            from_=from_,
            body=f'Your {first} code is {second}',
            to=to,
        )
        print(f'Sent message "{message.body}"')
    except FileNotFoundError as e:
        print(f"Failed to send notification: file {e.filename} not found")
    except TwilioRestException as e:
        print(f"Failed to send notification: {e.status}")
