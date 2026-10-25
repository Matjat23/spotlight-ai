"""App log: what happens inside spotlight.io, for debugging and the System page.

One JSON object per line in logs/app.log. The file rotates at 5 MB and the last 5 files are kept,
so the disk never fills up. The last 300 entries are also kept in memory for the System page.

Privacy rule: never log file names (they can contain children's names), captions, photos or the
words someone searched for. Only photo numbers, counts and timings.
"""
import json
import logging
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
RECENT = deque(maxlen=300)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "data", {}),
        }
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry)


class MemoryHandler(logging.Handler):
    def emit(self, record):
        RECENT.append(json.loads(self.format(record)))


def setup():
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("spotlight")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = JsonFormatter()
    for handler in (RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
                    MemoryHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


log = setup()


def info(event, **data):
    log.info(event, extra={"data": data})


def warning(event, **data):
    log.warning(event, extra={"data": data})


def error(event, **data):
    """Call inside an except block, the traceback is added automatically."""
    log.error(event, exc_info=True, extra={"data": data})
