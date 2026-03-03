"""
acl_filter.py — Retrieval-time Access Control List (ACL) filtering.

WHY THIS EXISTS:
━━━━━━━━━━━━━━
In a bank, authenticating a user (Phase 4.3 `auth.py`) is only half the
battle. You must also ensure they only "see" document chunks they are
authorized to see. This is "Retrieval ACLs."

If an Analyst (role: analyst) asks a question that would normally
retrieve internal strategy memos (doc_type: internal), those chunks
must be SILENTLY STRIPPED from the retrieval results before the LLM
ever sees them.

This prevents "privilege escalation" via RAG.

CLASSIFICATIONS:
  - handbook: Public (All roles)
  - guidance: Public (All roles)
  - policy: Restricted (Compliance, Admin)
  - internal: Confidential (Compliance, Admin)
  - hr: Strictly Confidential (Admin only)
"""

import logging
from typing import List
from langchain_core.documents import Document
from src.auth import UserRole

logger = logging.getLogger(__name__)

# ACL Map: Defines which roles can access which document types
# If a doc_type is not in this map, it defaults to public.
ACL_POLICY = {
    # Public types
    "handbook": [UserRole.ANALYST, UserRole.COMPLIANCE, UserRole.ADMIN],
    "guidance": [UserRole.ANALYST, UserRole.COMPLIANCE, UserRole.ADMIN],
    "mifid":    [UserRole.ANALYST, UserRole.COMPLIANCE, UserRole.ADMIN],
    "basel":    [UserRole.ANALYST, UserRole.COMPLIANCE, UserRole.ADMIN],
    
    # Restricted types
    "policy":   [UserRole.COMPLIANCE, UserRole.ADMIN],
    "internal": [UserRole.COMPLIANCE, UserRole.ADMIN],
    
    # Strictly Confidential
    "hr": [UserRole.ADMIN],
}


def filter_chunks_by_role(
    chunks: List[Document], 
    user_role: str
) -> tuple[List[Document], int]:
    """
    Filter retrieved chunks based on the user's role and document classification.

    Args:
        chunks: List of retrieved LangChain Document objects.
        user_role: The role string from the authenticated user context.

    Returns:
        A tuple of (filtered_chunks, drop_count).
    """
    filtered = []
    dropped = 0
    
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        
        # If doc_type has a specific ACL policy, check it
        if doc_type in ACL_POLICY:
            allowed_roles = ACL_POLICY[doc_type]
            if user_role in allowed_roles:
                filtered.append(chunk)
            else:
                dropped += 1
                logger.warning(
                    f"ACL DENIED: User (role={user_role}) blocked from chunk "
                    f"[id={chunk.metadata.get('chunk_id')}, type={doc_type}]"
                )
        else:
            # Default to allowing unknown types if not explicitly restricted
            filtered.append(chunk)
            
    if dropped > 0:
        logger.info(f"ACL Filter: Dropped {dropped}/{len(chunks)} chunks due to role restrictions.")
        
    return filtered, dropped
