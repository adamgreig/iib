import requests

PUSHOVER_USER = "gqRvM2dauS7ci1mSvHAPEMgc1dqhFr"
PUSHOVER_TOKEN = "62uoB6xLtjz6fgFwNgvWVCkL5dDhT2"
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_message(subject, message):
    data = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "title": subject,
        "message": message
    }
    requests.post(PUSHOVER_URL, data=data)
