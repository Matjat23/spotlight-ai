"""Staff accounts for spotlight.io, stored on this laptop in accounts.json.

Passwords are never stored, only a salted PBKDF2 hash (200,000 rounds of SHA-256), so even someone who
copies the file cannot read them. Two roles: admin (manages accounts, removes photos, sees the System
page) and staff (searches, edits, downloads and adds photos).
"""
import hashlib
import hmac
import json
import secrets
from datetime import datetime
from pathlib import Path

FILE = Path(__file__).parent / "accounts.json"
ROLES = ["staff", "admin"]
ROUNDS = 200_000
MIN_PASSWORD = 8


def _load():
    return json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}


def _save(users):
    FILE.write_text(json.dumps(users, indent=1), encoding="utf-8")


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ROUNDS).hex()


def users():
    """Accounts without their password hashes, for the Accounts page."""
    return {name: {k: v for k, v in u.items() if k not in ("salt", "hash")} for name, u in _load().items()}


def any_users():
    return bool(_load())


def add(name, password, role="staff"):
    """Returns an error message, or None when the account was made."""
    name = name.strip().lower()
    all_users = _load()
    if not name or not name.replace(".", "").replace("-", "").isalnum():
        return "Use letters and numbers for the user name"
    if name in all_users:
        return "That user name is taken"
    if len(password) < MIN_PASSWORD:
        return f"The password needs at least {MIN_PASSWORD} characters"
    if role not in ROLES:
        return "Unknown role"
    salt = secrets.token_hex(16)
    all_users[name] = {"role": role, "salt": salt, "hash": _hash(password, salt),
                       "created": datetime.now().isoformat(timespec="seconds")}
    _save(all_users)
    return None


def set_password(name, password):
    all_users = _load()
    if name not in all_users:
        return "No such user"
    if len(password) < MIN_PASSWORD:
        return f"The password needs at least {MIN_PASSWORD} characters"
    salt = secrets.token_hex(16)
    all_users[name].update(salt=salt, hash=_hash(password, salt))
    _save(all_users)
    return None


def remove(name):
    all_users = _load()
    all_users.pop(name, None)
    _save(all_users)


def check(name, password):
    """The role when the name and password match, else None. Constant-time compare."""
    u = _load().get(name.strip().lower())
    if u and hmac.compare_digest(u["hash"], _hash(password, u["salt"])):
        return u["role"]
    return None
