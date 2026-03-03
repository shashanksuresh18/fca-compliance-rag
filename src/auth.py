"""
auth.py — JWT authentication and role-based access control (RBAC).

WHY THIS EXISTS:
━━━━━━━━━━━━━━
A production RAG system in a bank cannot be "open." Different users
have different levels of access. For example:
  - An Analyst can query public FCA Handbook docs.
  - A Compliance Officer can query internal policy memos.
  - An Admin can trigger document ingestion.

This module:
  1. Validates JWT bearer tokens from the Authorization header.
  2. Extracts the user's role (admin, compliance, analyst).
  3. Rejects requests with invalid or expired tokens.
  4. Provides the user context to the rest of the RAG pipeline.

DESIGN:
  - Uses PyJWT for secure token verification.
  - Role-based: roles map to retrieval-time filters (ACLs).
  - Test/Demo mode: provides a utility to generate mock tokens.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

# Security scheme for FastAPI
reusable_oauth2 = HTTPBearer()

# Standard roles for bank RAG
class UserRole:
    ADMIN = "admin"           # Can ingest and query everything
    COMPLIANCE = "compliance" # Can query internal docs + handbook
    ANALYST = "analyst"       # Can only query handbook/guidance

class UserContext(BaseModel):
    user_id: str
    role: str
    tenant_id: str = "default"


def create_access_token(
    user_id: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Utility for generating JWT tokens for testing/demo.
    In a real bank, this happens in the Identity Provider (Azure AD/Okta).
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=8)
    
    to_encode = {
        "sub": user_id,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": "fca-rag-server"
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.api_secret_key, # Reusing api_secret_key as JWT secret for demo
        algorithm="HS256"
    )
    return encoded_jwt


async def get_current_user(
    auth: HTTPAuthorizationCredentials = Security(reusable_oauth2)
) -> UserContext:
    """
    FastAPI dependency to extract and validate the JWT from the request header.
    
    Usage in endpoint:
      @app.get("/query")
      def query(user: UserContext = Depends(get_current_user)): ...
    """
    token = auth.credentials
    
    # --- UI Simulation Mode (Phase 5) ---
    # Allow mock tokens from the UI for easy RBAC/ACL demonstration
    if token.startswith("MOCK_TOKEN_ROLE_"):
        role = token.replace("MOCK_TOKEN_ROLE_", "").lower()
        if role in [UserRole.ADMIN, UserRole.COMPLIANCE, UserRole.ANALYST]:
            return UserContext(user_id=f"ui_sim_{role}", role=role)

    try:
        payload = jwt.decode(
            token,
            settings.api_secret_key,
            algorithms=["HS256"],
            issuer="fca-rag-server"
        )
        user_id: str = payload.get("sub")
        role: str = payload.get("role")
        
        if user_id is None or role is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing subject or role claims."
            )
            
        return UserContext(user_id=user_id, role=role)
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT attempt: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate credentials.")


async def verify_admin(user: UserContext = Security(get_current_user)):
    """Dependency to restrict endpoints to Admin role only."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Admin role required for this operation."
        )
    return user
