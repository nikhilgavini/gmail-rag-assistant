import os
import os.path
import config

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def create_service(client_secret_file, api_name, api_version, scopes):
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = scopes
    
    creds = None
    token_path = config.TOKEN_FILE

    '''
    The file token.json stores the user's access and refresh tokens, and is
    created automatically when the authorization flow completes for the first
    time.
    '''
    # If the token exists, use it.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build(API_SERVICE_NAME, API_VERSION, credentials=creds, static_discovery=False)
        print(API_SERVICE_NAME, API_VERSION, 'service created successfully')
        return service
    except Exception as e:
        print(e)
        print(f'Failed to create service instance for {API_SERVICE_NAME}')
        os.remove(token_path)
        return None


def init_gmail_service(client_file, api_name='gmail', api_version='v1', scopes=SCOPES):
    return create_service(client_file, api_name, api_version, scopes)

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def _extract_body(payload):
    body = '<Text body not available>'
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'multipart/alternative':
                for subpart in part['parts']:
                    if subpart['mimeType'] == 'text/plain' and 'data' in subpart['body']:
                        body = base64.urlsafe_b64decode(subpart['body']['data']).decode('utf-8')
                        break
            elif part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
    elif 'body' in payload and 'data' in payload['body']:
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return body


def _aggregate_email_text(subject, sender, msg_date, body):
    return f"Subject: {subject}\nFrom: {sender}\nDate: {msg_date}\n\n{body}"


def get_list_of_folders(service):
    # Folders we don't care about
    remove_labels = [
      'YELLOW_STAR',
      'TRASH',
      'Notes',
      'CHAT',
      'DRAFT',
      'CATEGORY_PROMOTIONS',
      'CATEGORY_FORUMS',
      'CATEGORY_PERSONAL',
      'CATEGORY_UPDATES',
      'CATEGORY_SOCIAL'
  ]

    results = service.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])

    if not labels:
      print("No labels found.")
      return []
    else:
      labels = list({l['name'] for l in labels if 'name' in l})
      good_labels = [x for x in labels if x not in remove_labels]
      return good_labels

###############################################################################
# DATA INGEST WITH GMAIL API
###############################################################################
def get_email_messages(service, user_id='me', label_ids=None, folder_name='INBOX', max_results=5, query=None):
    messages = []
    next_page_token = None

    if folder_name: # If a folder name is provided, we need to get the label ID for the folder
        label_results = service.users().labels().list(userId=user_id).execute()
        labels = label_results.get('labels', [])
        folder_label_id = next((label['id'] for label in labels if label['name'].lower() == folder_name.lower()), None)
        if folder_label_id:
            if label_ids:
                label_ids.append(folder_label_id)
            else:
                label_ids = [folder_label_id]
        else:
            raise ValueError(f"Folder '{folder_name}' not found.")
    
    while True: # Continue fetching messages until we have reached the max_results or there are no more messages
        result = service.users().messages().list(
            userId = user_id,
            labelIds = label_ids,
            q = query,
            maxResults = min(500, max_results - len(messages)) if max_results else 500, # This method can only fetch 500 messages per API call
            pageToken = next_page_token
        ).execute()

        messages.extend(result.get('messages', []))
        next_page_token = result.get('nextPageToken')

        if not next_page_token or (max_results and len(messages) >= max_results):
            break
    
    return messages[:max_results] if max_results else messages  # Ensures we return exactly the number of messages requested


def get_email_message_details(service, msg_id):
    message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = message['payload']
    headers = payload.get('headers', [])    # Contain important metadata about the email

    subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), None)
    if not subject:
        subject = message.get('subject', 'No subject')
    
    sender = next((header['value'] for header in headers if header['name'] == 'From'), 'No sender')
    recipients = next((header['value'] for header in headers if header['name'] == 'To'), 'No recipients')
    snippet = message.get('snippet', 'No snippet')
    has_attachments = any(part.get('filename') for part in payload.get('parts', []) if part.get('filename'))
    date = next((header['value'] for header in headers if header['name'] == 'Date'), 'No date')
    star = message.get('labelIds', []).count('STARRED') > 0
    label = ', '.join(message.get('labelIds', []))

    body = _extract_body(payload)

    text = _aggregate_email_text(subject, sender, date, body)

    return {
        'type': 'email',
        'source': subject,
        'text': text,
        'metadata': {
            'id': msg_id,
            'date': date,
            'sender': sender
        }
    }