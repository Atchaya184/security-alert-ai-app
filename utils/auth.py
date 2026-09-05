"""
auth.py
-----------------------------------------
Minimal shared-token authorization for rollback and high-impact actions.

This project has no user database or login system, so a full auth
overhaul is out of scope. This module adds the smallest mechanism that
actually blocks unauthenticated requests: a set of bearer tokens, each
mapped to an analyst display name.

A protected request must present a valid token via either:

    Authorization: Bearer <token>
    X-Auth-Token: <token>

Tokens are configured via the ANALYST_AUTH_TOKENS environment variable:
a comma-separated list of `token:name` pairs, e.g.

    ANALYST_AUTH_TOKENS="s3cr3t-1:Alice,s3cr3t-2:Bob"

If that variable isn't set, a single default development token is used
(see DEFAULT_TOKENS below) so the app keeps working out of the box in a
demo/dev environment. CHANGE THIS before any real deployment.

Tests: the existing automated test suite calls these routes directly via
Flask's test client and never sends auth headers. Rather than rewrite
every existing test, `require_auth` is skipped whenever
`app.config["TESTING"]` is True -- the same flag Flask itself already
uses to change other behaviors (e.g. error propagation). This keeps the
pre-existing tests passing unmodified. New tests that specifically verify
authorization flip TESTING off for that one request (see
tests/test_app.py).
-----------------------------------------
"""

import os
from functools import wraps
from flask import request, jsonify, current_app, g

# Default token for local/dev use only. Override with ANALYST_AUTH_TOKENS
# in any real deployment.
DEFAULT_TOKENS = {
    "dev-analyst-token-CHANGE-ME": "Analyst",
}


def _load_tokens():
    raw = os.environ.get("ANALYST_AUTH_TOKENS")
    if not raw:
        return dict(DEFAULT_TOKENS)
    tokens = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" in pair:
            token, name = pair.split(":", 1)
        else:
            token, name = pair, "Analyst"
        token = token.strip()
        name = name.strip() or "Analyst"
        if token:
            tokens[token] = name
    return tokens or dict(DEFAULT_TOKENS)


# Loaded once at import time (env var is read at process startup, matching
# how the rest of the app reads its configuration).
TOKENS = _load_tokens()


def _extract_token():
    header = request.headers.get("X-Auth-Token")
    if header:
        return header.strip()
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def current_analyst_name(default="Analyst"):
    """Name of the authenticated caller, if `require_auth` ran for this
    request; falls back to `default` otherwise (e.g. in TESTING mode)."""
    return getattr(g, "analyst_name", default)


def require_auth(f):
    """Protect a Flask view: require a valid analyst auth token.

    Skipped when app.config['TESTING'] is True so the pre-existing pytest
    suite (which never sends auth headers) is unaffected. See module
    docstring.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if current_app.config.get("TESTING"):
            g.analyst_name = "Analyst"
            return f(*args, **kwargs)

        token = _extract_token()
        analyst_name = TOKENS.get(token) if token else None
        if not token or analyst_name is None:
            return jsonify({
                "error": "Unauthorized: a valid analyst auth token is required "
                         "(send it as 'X-Auth-Token: <token>' or "
                         "'Authorization: Bearer <token>')."
            }), 401
        g.analyst_name = analyst_name
        return f(*args, **kwargs)
    return wrapper
