from datetime import datetime
import pytz
from config import TIMEZONE

def now_iso() -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).isoformat(timespec="seconds")
