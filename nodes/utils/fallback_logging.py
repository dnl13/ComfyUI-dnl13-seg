import logging
# Erstelle einen benutzerdefinierten Formatter für farbige Protokolle
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',   # Red
        'CRITICAL': '\033[1;31m'  # Bright Red
    }

    def format(self, record):
        log_message = super().format(record)
        log_level_color = self.COLORS.get(record.levelname, '')
        return f"{log_level_color}{log_message}\033[0m"

# Konfiguriere den Logger
logger = logging.getLogger(__name__)


# Erstelle einen StreamHandler mit dem benutzerdefinierten Formatter
handler = logging.StreamHandler()
formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Füge den Handler zum Logger hinzu
logger.addHandler(handler)