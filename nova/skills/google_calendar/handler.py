"""
nova/skills/google_calendar/handler.py
Google Calendar skill handler. Uses OAuth2 installed app flow (one-time,
offline via scripts/get_google_token.py) and the Google Calendar API to
list, create, update and delete events.

Configuration (set these in .env or environment):
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_TOKEN_PATH (optional, default: ./data/google_calendar_token.json)
  GOOGLE_CALENDAR_TIMEZONE (optional, e.g. Europe/London)

This skill exposes a `skill` instance (BaseSkill subclass) discovered by the
ToolRouter.
"""
import datetime
import logging
import os
import re
from typing import Optional

from dateutil import tz

from nova.skills.base_skill import BaseSkill
from nova.skills.google_calendar.tools import TOOLS

logger = logging.getLogger("nova.skills.google_calendar")

# Import Google APIs lazily to avoid hard failure when deps are not installed
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover - optional dependency
    Request = None
    Credentials = None
    build = None


SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Real Google Calendar event IDs are opaque lowercase base32-ish strings
# (letters a-v and digits), typically 26 chars, no spaces. Anything that
# doesn't look like this is almost certainly the model hallucinating a
# placeholder (e.g. "exam", "DB exam on 2026-09-04") rather than a real ID -
# treat those as search queries instead of failing with a 404.
_REAL_EVENT_ID_PATTERN = re.compile(r"^[a-v0-9]{20,}$")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _token_path() -> str:
    return _env("GOOGLE_TOKEN_PATH", os.path.join("data", "google_calendar_token.json"))


def _load_credentials() -> Optional[object]:
    """Load the cached refresh token and silently refresh it if needed.
    Never runs an interactive flow - that happens once, offline, via
    scripts/get_google_token.py.
    """
    if Credentials is None:
        raise RuntimeError(
            "Google API libraries not installed. Add google-api-python-client, "
            "google-auth-oauthlib, google-auth-httplib2 to requirements and install them."
        )

    token_file = _token_path()
    if not os.path.exists(token_file):
        raise RuntimeError(
            f"No Google Calendar token found at {token_file}. Run "
            "scripts/get_google_token.py on Windows first, then restart NOVA."
        )

    try:
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    except Exception as exc:
        raise RuntimeError(f"Failed to load credentials from {token_file}: {exc}") from exc

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        return creds

    raise RuntimeError(
        "Google Calendar token exists but is invalid and has no refresh token. "
        "Re-run scripts/get_google_token.py."
    )


def _build_service():
    creds = _load_credentials()
    if not creds:
        raise RuntimeError("Unable to obtain Google credentials.")
    service = build("calendar", "v3", credentials=creds)
    return service


def _start_of_day_rfc3339(tz_name: Optional[str] = None, date: Optional[datetime.date] = None) -> str:
    tzinfo = tz.gettz(tz_name) if tz_name else datetime.datetime.now().astimezone().tzinfo
    base = date or datetime.datetime.now(tz=tzinfo).date()
    start = datetime.datetime(base.year, base.month, base.day, 0, 0, 0, tzinfo=tzinfo)
    return start.isoformat()


def _end_of_day_rfc3339(tz_name: Optional[str] = None, date: Optional[datetime.date] = None) -> str:
    tzinfo = tz.gettz(tz_name) if tz_name else datetime.datetime.now().astimezone().tzinfo
    base = date or datetime.datetime.now(tz=tzinfo).date()
    end = datetime.datetime(base.year, base.month, base.day, 23, 59, 59, tzinfo=tzinfo)
    return end.isoformat()


def _looks_like_real_event_id(value: str) -> bool:
    return bool(_REAL_EVENT_ID_PATTERN.match(value.strip()))


def _local_now() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


def _format_event_time(start_value: Optional[str], end_value: Optional[str] = None) -> str:
    """Return a human-friendly event time string without raw RFC3339 noise.

    We keep the original event timezone offset when Google already supplied one;
    only apply an explicit user timezone override when configured. Converting a
    time to the container/server timezone (for example UTC) can shift the visible
    time by one or more hours and cause events to appear earlier than they really are.
    """
    if not start_value:
        return "time not specified"

    tz_name = _env("GOOGLE_CALENDAR_TIMEZONE")
    target_tz = tz.gettz(tz_name) if tz_name else None

    try:
        if "T" in start_value:
            start_dt = datetime.datetime.fromisoformat(start_value)
            if start_dt.tzinfo is not None and target_tz is not None:
                start_dt = start_dt.astimezone(target_tz)

            if end_value and "T" in end_value:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_value)
                    if end_dt.tzinfo is not None and target_tz is not None:
                        end_dt = end_dt.astimezone(target_tz)
                    return f"{start_dt.strftime('%I:%M %p').lstrip('0')} to {end_dt.strftime('%I:%M %p').lstrip('0')}"
                except ValueError:
                    pass

            return start_dt.strftime('%I:%M %p').lstrip('0')

        try:
            start_date = datetime.date.fromisoformat(start_value)
            return start_date.strftime('%b %d')
        except ValueError:
            return start_value
    except ValueError:
        return start_value


def _format_event_line(ev: dict) -> str:
    summary = ev.get("summary", "(no title)")
    start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date"))
    end = ev.get("end", {}).get("dateTime", ev.get("end", {}).get("date"))
    time_text = _format_event_time(start, end)

    if start and "T" in str(start):
        return f"- {summary} at {time_text}"
    if start and "-" in str(start):
        return f"- {summary} on {time_text}"
    return f"- {summary}"


class GoogleCalendarSkill(BaseSkill):
    def __init__(self):
        self._service = None

    @property
    def tools(self) -> list[dict]:
        return TOOLS

    def _ensure_service(self):
        if self._service is None:
            self._service = _build_service()

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "gc_list_today":
                return self._list_today(arguments)

            if tool_name == "gc_find":
                return self._find(arguments)

            if tool_name == "gc_create_event":
                return self._create_event(arguments)

            if tool_name == "gc_update_event":
                return self._update_event(arguments)

            if tool_name == "gc_delete_event":
                return self._delete_event(arguments)

            return "Unknown Google Calendar command."
        except Exception as exc:
            logger.exception("Error executing %s", tool_name)
            # Keep the raw exception in logs for debugging, but return a simple user
            # sentence so the interface does not leak API details or stack traces.
            raise RuntimeError(f"I couldn't complete that calendar task.") from exc

    def _list_today(self, arguments: dict) -> str:
        calendar_id = arguments.get("calendar_id", "primary")
        tz_name = _env("GOOGLE_CALENDAR_TIMEZONE")
        self._ensure_service()
        time_min = _start_of_day_rfc3339(tz_name)
        time_max = _end_of_day_rfc3339(tz_name)
        events_result = (
            self._service.events()
            .list(calendarId=calendar_id, timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime")
            .execute()
        )
        items = events_result.get("items", [])
        if not items:
            return "You do not have any events scheduled for today."
        if len(items) == 1:
            ev = items[0]
            summary = ev.get("summary", "(no title)")
            start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date"))
            end = ev.get("end", {}).get("dateTime", ev.get("end", {}).get("date"))
            time_text = _format_event_time(start, end)
            return f"Today you have {summary} at {time_text} on your calendar."
        lines = [_format_event_line(ev) for ev in items]
        return "Today you have:\n" + "\n".join(lines)

    def _find(self, arguments: dict) -> str:
        query = arguments.get("query")
        time_min = arguments.get("time_min") or _start_of_day_rfc3339()
        time_max = arguments.get("time_max") or _end_of_day_rfc3339()
        calendar_id = arguments.get("calendar_id", "primary")
        self._ensure_service()
        events_result = (
            self._service.events()
            .list(calendarId=calendar_id, q=query, timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime")
            .execute()
        )
        items = events_result.get("items", [])
        if not items:
            return f"No events found matching '{query}'."
        lines = [_format_event_line(ev) for ev in items]
        return "I found these events:\n" + "\n".join(lines)

    def _resolve_event(self, arguments: dict) -> tuple[Optional[dict], Optional[str]]:
        """
        Resolves `arguments` (which may contain a real event_id, OR a query/
        date describing the event by name) down to a single matching event
        dict, or returns (None, error_message) if it can't.

        This is the fix for the model hallucinating placeholder event_ids
        (e.g. "exam") instead of the real Google-issued ID it was never
        given: if event_id doesn't look like a real ID, or is absent,
        fall back to a calendar search using query/date instead.
        """
        calendar_id = arguments.get("calendar_id", "primary")
        event_id = (arguments.get("event_id") or "").strip()

        # Case 1: a plausible real event_id was provided - try it directly.
        if event_id and _looks_like_real_event_id(event_id):
            try:
                ev = self._service.events().get(calendarId=calendar_id, eventId=event_id).execute()
                return ev, None
            except Exception:
                logger.info("event_id %r looked real but was not found; falling back to search.", event_id)

        # Case 2: search by query (or by whatever was put in event_id if it
        # wasn't a real ID - the model likely meant it as a description).
        query = arguments.get("query") or (event_id if event_id else None)
        if not query:
            return None, "I don't know which event you mean - please tell me its title (and date, if helpful)."

        tz_name = _env("GOOGLE_CALENDAR_TIMEZONE")
        date_str = arguments.get("date")
        if date_str:
            try:
                target_date = datetime.date.fromisoformat(date_str)
            except ValueError:
                target_date = None
        else:
            target_date = None

        if target_date:
            time_min = _start_of_day_rfc3339(tz_name, target_date)
            time_max = _end_of_day_rfc3339(tz_name, target_date)
        else:
            # No date given - search a wide window (30 days back to 180
            # days forward) rather than just "today", since the event may
            # be in the past or future.
            now = datetime.datetime.now(tz=tz.gettz(tz_name) if tz_name else None)
            time_min = (now - datetime.timedelta(days=30)).isoformat()
            time_max = (now + datetime.timedelta(days=180)).isoformat()

        events_result = (
            self._service.events()
            .list(calendarId=calendar_id, q=query, timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime")
            .execute()
        )
        items = events_result.get("items", [])

        if not items:
            when = f" on {date_str}" if date_str else ""
            return None, f"I couldn't find an event matching '{query}'{when}."

        if len(items) > 1:
            lines = [_format_event_line(ev) for ev in items]
            return None, (
                f"I found multiple events matching '{query}' - please be more specific "
                f"(e.g. include the date):\n" + "\n".join(lines)
            )

        return items[0], None

    def _create_event(self, arguments: dict) -> str:
        summary = arguments.get("summary")
        if not summary:
            return "Missing event summary."
        start = arguments.get("start")
        if not start:
            return "Missing event start time. Provide an RFC3339 timestamp or a date-time string."

        try:
            start_dt = datetime.datetime.fromisoformat(start)
        except ValueError as exc:
            return f"Invalid start time '{start}'. Use an RFC3339 timestamp like 2026-08-22T22:00:00+01:00."

        end = arguments.get("end")
        duration = arguments.get("duration_minutes")
        if end:
            try:
                end_dt = datetime.datetime.fromisoformat(end)
            except ValueError:
                return f"Invalid end time '{end}'. Use an RFC3339 timestamp like 2026-08-22T23:30:00+01:00."

            # Many natural-language requests use 'midnight' or 'till midnight' for
            # overnight events, yielding an end time like 00:00 of the same date.
            # Treat that as the next day when the start is later in the evening.
            if end_dt <= start_dt and end_dt.hour <= 3:
                end_dt = end_dt + datetime.timedelta(days=1)
                end = end_dt.isoformat()
        elif duration:
            end_dt = start_dt + datetime.timedelta(minutes=int(duration))
            end = end_dt.isoformat()
        else:
            return "Provide either end or duration_minutes to create an event."

        event_body = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
        if arguments.get("description"):
            event_body["description"] = arguments.get("description")
        if arguments.get("location"):
            event_body["location"] = arguments.get("location")

        calendar_id = arguments.get("calendar_id", "primary")
        self._ensure_service()
        ev = self._service.events().insert(calendarId=calendar_id, body=event_body).execute()
        time_text = _format_event_time(start, end)
        return f"Created {summary} for {time_text}."

    def _update_event(self, arguments: dict) -> str:
        self._ensure_service()
        ev, error = self._resolve_event(arguments)
        if error:
            return error

        calendar_id = arguments.get("calendar_id", "primary")
        updated = False
        for field in ("summary", "description"):
            if field in arguments and arguments[field] is not None:
                ev[field] = arguments[field]
                updated = True
        if "start" in arguments and arguments.get("start"):
            ev["start"] = {"dateTime": arguments.get("start")}
            updated = True
        if "end" in arguments and arguments.get("end"):
            ev["end"] = {"dateTime": arguments.get("end")}
            updated = True
        if not updated:
            return "No updatable fields provided."
        ev = self._service.events().update(calendarId=calendar_id, eventId=ev["id"], body=ev).execute()
        summary = ev.get("summary", "event")
        start = ev.get("start", {}).get("dateTime")
        end = ev.get("end", {}).get("dateTime")
        time_text = _format_event_time(start, end)
        if time_text and time_text != "time not specified":
            return f"Updated {summary} to {time_text}."
        return f"Updated {summary}."

    def _delete_event(self, arguments: dict) -> str:
        self._ensure_service()
        ev, error = self._resolve_event(arguments)
        if error:
            return error

        calendar_id = arguments.get("calendar_id", "primary")
        summary = ev.get("summary", "(no title)")
        self._service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
        return f"Deleted {summary} from your calendar."


# Expose skill instance discovered by ToolRouter
skill = GoogleCalendarSkill()