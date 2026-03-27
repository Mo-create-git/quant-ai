"""
auth.py
───────
Real login & signup stored in SQLite.
Passwords are hashed using SHA-256.
Returns a simple token (email) the frontend stores in localStorage.
"""

import hashlib
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.database import get_connection

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


# ── Helper ─────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/signup")
def signup(req: SignupRequest):
    """Register a new user."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)",
            (req.first_name, req.last_name, req.email, hash_password(req.password))
        )
        conn.commit()
        return JSONResponse({
            "success": True,
            "message": "Account created!",
            "user": {"email": req.email, "first_name": req.first_name, "last_name": req.last_name}
        })
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(400, "Email already registered. Please log in.")
        raise HTTPException(500, str(e))
    finally:
        conn.close()


@router.post("/login")
def login(req: LoginRequest):
    """Log in an existing user."""
    conn = get_connection()
    try:
        user = conn.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (req.email, hash_password(req.password))
        ).fetchone()

        if not user:
            raise HTTPException(401, "Invalid email or password.")

        return JSONResponse({
            "success": True,
            "message": "Logged in!",
            "user": {
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"]
            }
        })
    finally:
        conn.close()


@router.get("/users")
def list_users():
    """List all registered users (for demo/admin purposes)."""
    conn = get_connection()
    try:
        users = conn.execute(
            "SELECT id, first_name, last_name, email, created_at FROM users"
        ).fetchall()
        return {"users": [dict(u) for u in users], "total": len(users)}
    finally:
        conn.close()
