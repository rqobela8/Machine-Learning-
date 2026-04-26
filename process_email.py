import glob, re
from email import policy
from email.parser import BytesParser
from typing import List
from bs4 import BeautifulSoup

from message import Message


def extract_email_text(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # ----------------------------
    # Safe header extraction
    # ----------------------------
    subject = msg.get("subject") or ""
    sender = msg.get("from") or ""
    to = msg.get("to") or ""
    date = msg.get("date") or ""

    body = ""

    # ----------------------------
    # Helper: clean HTML if needed
    # ----------------------------
    def clean_html(text):
        return BeautifulSoup(text, "html5lib").get_text(separator=" ", strip=True)

    # ----------------------------
    # Extract body safely
    # ----------------------------
    if msg.is_multipart():
        # 1. Prefer plain text
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")

            # skip attachments
            if "attachment" in disposition:
                continue

            if content_type == "text/plain":
                try:
                    body = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body = payload.decode(charset, errors="replace")
                    except LookupError:
                        body = payload.decode("utf-8", errors="replace")
                break

        # 2. Fallback to HTML
        if not body:
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition") or "")

                if "attachment" in disposition:
                    continue

                if content_type == "text/html":
                    try:
                        html = part.get_content()
                    except Exception:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            html = payload.decode(charset, errors="replace")
                        except LookupError:
                            html = payload.decode("utf-8", errors="replace")

                    body = clean_html(html)
                    break

    else:
        try:
            body = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            try:
                body = payload.decode(charset, errors="replace")
            except LookupError:
                body = payload.decode("utf-8", errors="replace")

    # ----------------------------
    # Final cleanup
    # ----------------------------
    body = subject + "\n" + body.strip()
    return {
        "subject": subject,
        "from": sender,
        "to": to,
        "date": date,
        "body": body
    }


def build_messages() -> List[Message]|None:
    """
    Builds Message objects and add to List[Message]

    :return: List[Message]
    """
    files_path = 'spam_data/*/*' #File path for every file in data folder
    data: List[Message] = []

    #glob.glob returns every filename that matches the wildcarded path
    for filename in glob.glob(files_path):
        is_spam = "ham" not in filename  #If file is in Spam folder is_spam = True
        email_parts_dir = extract_email_text(filename)
        data.append(Message(email_parts_dir["body"], is_spam))

    return data

