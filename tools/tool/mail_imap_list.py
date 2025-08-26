import imaplib
import email
from email.header import decode_header
from typing import Dict, Any, List
from .base import ToolPlugin, ToolContext

def _decode(s):
    if s is None:
        return ""
    if isinstance(s, bytes):
        try:
            return s.decode("utf-8", "ignore")
        except Exception:
            return s.decode("latin1", "ignore")
    return str(s)

class MailImapList(ToolPlugin):
    name = "mail.imap_list"
    description = "IMAP으로 최근 메일 헤더 요약(읽기 전용)"
    input_schema = {
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "username": {"type": "string"},
            "password": {"type": "string"},
            "mailbox": {"type": "string", "default": "INBOX"},
            "max_count": {"type": "integer", "default": 10}
        },
        "required": ["host", "username", "password"],
        "additionalProperties": False
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        host = args["host"]; user = args["username"]; pw = args["password"]
        mailbox = args.get("mailbox", "INBOX")
        max_count = int(args.get("max_count", 10))

        m = imaplib.IMAP4_SSL(host)
        try:
            m.login(user, pw)
            m.select(mailbox)
            typ, data = m.search(None, "ALL")
            if typ != "OK":
                return self.fail("search failed", mailbox=mailbox)

            ids = data[0].split()
            ids = ids[-max_count:] if ids else []
            out: List[Dict[str, Any]] = []

            for i in reversed(ids):
                typ, msg_data = m.fetch(i, "(RFC822.HEADER)")
                if typ != "OK":
                    continue
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                def _dh(v):
                    parts = decode_header(msg.get(v))
                    s = ""
                    for text, enc in parts:
                        if isinstance(text, bytes):
                            s += text.decode(enc or "utf-8", "ignore")
                        else:
                            s += text
                    return s

                out.append({
                    "from": _dh("From"),
                    "subject": _dh("Subject"),
                    "date": _dh("Date"),
                })
            return self.ok(messages=out)
        finally:
            try:
                m.logout()
            except Exception:
                pass
