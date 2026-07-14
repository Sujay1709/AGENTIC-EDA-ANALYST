"""
Lightweight, env-based login gate for the Streamlit app.

Credentials come from environment variables so nothing is hard-coded in the
repo:

    APP_USERNAME   (default: "admin")
    APP_PASSWORD   (default: "admin")   # change this in production!

This is intentionally simple — a single shared login suitable for a local /
internal tool. For multi-user auth with hashed passwords, swap this module for
`streamlit-authenticator`; the rest of the app only depends on
`require_login()` and `logout()`.
"""
import hmac
import os

import streamlit as st


def _expected_credentials() -> tuple[str, str]:
    """Read the expected username/password from the environment."""
    return (
        os.getenv("APP_USERNAME", "admin"),
        os.getenv("APP_PASSWORD", "admin"),
    )


def _credentials_ok(username: str, password: str) -> bool:
    """Constant-time comparison so we don't leak timing information."""
    exp_user, exp_pass = _expected_credentials()
    user_ok = hmac.compare_digest(username or "", exp_user)
    pass_ok = hmac.compare_digest(password or "", exp_pass)
    return user_ok and pass_ok


def _render_login_form() -> None:
    """Draw a centered login card and validate on submit."""
    # Hide sidebar/header while the user is logged out.
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
          [data-testid="stAppViewContainer"] { background: #121C30; }
          .login-title { color: #FFFFFF; font-size: 26px; font-weight: 800;
                         letter-spacing: -.02em; margin-bottom: 4px; }
          .login-title span { color: #2A8C7E; }
          .login-sub { color: #98A2AE; font-size: 13.5px; margin-bottom: 22px; }
          div[data-testid="stForm"] {
              background: #1C2C48; border: 1px solid rgba(42,140,126,.25);
              border-radius: 12px; padding: 28px 26px;
          }
          .stButton > button, .stFormSubmitButton > button {
              background: #2A8C7E !important; color: #fff !important;
              border: none !important; border-radius: 6px !important;
              font-weight: 600 !important; width: 100%;
          }
          .stFormSubmitButton > button:hover { background: #38B2A2 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown(
            '<div class="login-title">Agentic <span>EDA</span> Pipeline</div>'
            '<div class="login-sub">Sign in to continue</div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input(
                "Password", type="password", autocomplete="current-password"
            )
            submitted = st.form_submit_button("Sign in")

        if submitted:
            if _credentials_ok(username, password):
                st.session_state.authenticated = True
                st.session_state.auth_user = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")

        exp_user, exp_pass = _expected_credentials()
        if exp_user == "admin" and exp_pass == "admin":
            st.caption(
                "⚠️ Using default credentials (admin / admin). Set `APP_USERNAME` "
                "and `APP_PASSWORD` environment variables before deploying."
            )


def require_login() -> None:
    """
    Gate the app behind a login screen.

    Call this once, right after `st.set_page_config`. If the user is not
    authenticated it renders the login form and halts the script with
    `st.stop()`, so nothing below it runs until the user signs in.
    """
    if st.session_state.get("authenticated"):
        return
    _render_login_form()
    st.stop()


def logout() -> None:
    """Clear the auth flag and rerun back to the login screen."""
    for key in ("authenticated", "auth_user"):
        st.session_state.pop(key, None)
    st.rerun()
