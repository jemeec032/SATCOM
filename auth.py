"""
Simple login system for the Satcom RF Impairment Simulator, with
account creation, backed by a local JSON file on disk.

IMPORTANT: This is a DEMO-GRADE login, not real production security.
- Passwords are hashed (SHA-256 + per-user salt) before being stored,
  which is better than plaintext, but this still has none of the
  hardening a real auth system needs (rate limiting, breach detection,
  proper secret management, HTTPS enforcement, etc.).
- The user database is a plain JSON file (users.json) sitting next to
  this script. Anyone with file access to the machine can read it.

This is fine for a personal/mentor-facing demo running on your own
machine or an offline institute PC. Do NOT reuse this pattern for
anything handling real user accounts or sensitive data.
"""

import json
import os
import hashlib
import secrets

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_account(username: str, password: str):
    """
    Register a new user. Returns (success: bool, message: str).
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    salt = secrets.token_hex(8)
    users[username] = {"salt": salt, "hash": _hash_password(password, salt)}
    _save_users(users)
    return True, "Account created. You can now log in."


def check_credentials(username: str, password: str) -> bool:
    """Return True if username/password match a stored account."""
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return False
    return _hash_password(password, entry["salt"]) == entry["hash"]
