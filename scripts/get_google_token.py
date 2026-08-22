"""
scripts/get_google_token.py

Manual-code-paste OAuth authorization. Avoids the local-server callback
flow entirely (it was failing silently in this environment). Google will
redirect your browser to a localhost page that shows an error - that is
expected. The authorization code will still be visible in the browser's
address bar on that page; copy the full URL and paste it back here when
prompted.
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ModuleNotFoundError as exc:
    print("Missing Google OAuth dependencies. Install them with: python3 -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

SCOPES = ["https://www.googleapis.com/auth/calendar"]
REDIRECT_URI = "http://localhost:8080/"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    client_config = {
        "installed": {
            "client_id": _env("GOOGLE_CLIENT_ID"),
            "client_secret": _env("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    configured = os.environ.get("GOOGLE_TOKEN_PATH", os.path.join("persistent_memory", "credentials", "google_calendar_token.json"))
    token_path = Path(configured)
    if not token_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path.cwd() / token_path,
            project_root / token_path,
            project_root / "data" / token_path.name,
        ]
        for candidate in candidates:
            if candidate.exists() or str(candidate).endswith(str(token_path)):
                token_path = candidate
                break
        else:
            token_path = project_root / token_path

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    flow.redirect_uri = REDIRECT_URI

    auth_url, _state = flow.authorization_url(access_type="offline", prompt="consent")

    print("=" * 70)
    print("1. Open this URL in your browser:\n")
    print(auth_url)
    print("\n2. Sign in and click Allow.")
    print("3. The browser will land on a localhost:8080 page showing an error.")
    print("   THAT IS EXPECTED - ignore it, do not close the tab yet.")
    print("4. Copy the FULL URL from the address bar and paste it below.")
    print("=" * 70)

    pasted = input("\nPaste the full redirected URL here: ").strip()

    parsed = urlparse(pasted)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]

    if not code:
        print("\nERROR: no 'code' parameter found in what you pasted.")
        sys.exit(1)

    print("\nExchanging authorization code for a token...")
    flow.fetch_token(code=code)
    creds = flow.credentials

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")

    print(f"\nSUCCESS - token saved to {token_path.resolve()}")
    print("This is the default secure storage path for Google Calendar OAuth tokens.")
    print(f"Has refresh_token: {bool(creds.refresh_token)}")


if __name__ == "__main__":
    main()