from llamaherd import proxy


def test_session_cookie_is_secure_for_https_requests():
    token = proxy._secure_cookie_context.set(True)
    try:
        cookie = proxy._session_cookie_for_response("session-id", 300)
    finally:
        proxy._secure_cookie_context.reset(token)
    assert "; Secure" in cookie
    assert "; HttpOnly" in cookie
    assert "; SameSite=Lax" in cookie


def test_session_cookie_allows_plain_http_development():
    token = proxy._secure_cookie_context.set(False)
    try:
        cookie = proxy._session_cookie_for_response("session-id", 300)
    finally:
        proxy._secure_cookie_context.reset(token)
    assert "; Secure" not in cookie
