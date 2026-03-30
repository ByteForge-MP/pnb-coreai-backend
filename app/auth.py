import json
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

security = HTTPBearer(auto_error=False)
active_tokens = {}


class LoginRequest(BaseModel):
    username: str
    password: str


DEFAULT_USERS = {
    "admin": {
        "password": "admin123",
        "name": "Admin User",
    },
    "mayank": {
        "password": "mayank123",
        "name": "Mayank Prakash",
    },
    "john": {
        "password": "john123",
        "name": "John Doe",
    },
}


def _load_users():
    raw_users = os.getenv("AUTH_USERS", "").strip()

    if not raw_users:
        return DEFAULT_USERS

    try:
        parsed = json.loads(raw_users)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid AUTH_USERS JSON: {exc}") from exc

    users = {}

    for username, details in parsed.items():
        if not isinstance(details, dict):
            continue

        password = str(details.get("password", "")).strip()
        name = str(details.get("name", username)).strip() or username

        if username and password:
            users[username] = {
                "password": password,
                "name": name,
            }

    return users or DEFAULT_USERS


def _bad_credentials():
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


def login_user(payload: LoginRequest):
    username = payload.username.strip()
    password = payload.password.strip()
    users = _load_users()

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="username and password are required",
        )

    user_record = users.get(username)

    if user_record is None or password != user_record["password"]:
        _bad_credentials()

    token = secrets.token_urlsafe(32)
    user = {
        "name": user_record["name"],
        "username": username,
    }
    active_tokens[token] = user

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = credentials.credentials
    user = active_tokens.get(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return user
