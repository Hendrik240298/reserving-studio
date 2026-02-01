from __future__ import annotations

import yaml
import logging
from pathlib import Path
from datetime import datetime
import re


class ConfigManager:
    def __init__(self, config: dict, config_path: Path | None = None):
        self._config_path = Path(config_path) if config_path else None
        self._config = config
        self._init_paths(config)
        self._init_properties(config)
        self._init_sessions(config)

    @classmethod
    def from_yaml(cls, file_name):
        config = cls._read_yaml(file_name)
        return cls(config, config_path=Path(file_name))

    @staticmethod
    def _read_yaml(file_path):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Config file {file_path} does not exist.")
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _init_paths(self, config):
        self._RESULTS = config["paths"]["results"]
        self._PLOTS = config["paths"]["plots"]
        self._DATA = config["paths"]["data"]
        self._SESSIONS = config.get("paths", {}).get("sessions", "sessions/")

        if not Path(self._RESULTS).exists():
            Path(self._RESULTS).mkdir(parents=True, exist_ok=True)
            Path(self._RESULTS + "dashboard/").mkdir(parents=True, exist_ok=True)
        if not Path(self._PLOTS).exists():
            Path(self._PLOTS).mkdir(parents=True, exist_ok=True)
        if not Path(self._DATA).exists():
            Path(self._DATA).mkdir(parents=True, exist_ok=True)
        if not Path(self._SESSIONS).exists():
            Path(self._SESSIONS).mkdir(parents=True, exist_ok=True)

    def _init_properties(self, config):
        first_date = config.get("first date", None)
        if first_date is None or first_date == "":
            self._first_date = 1900  # set a very old date to include all claims
        elif len(str(int(first_date))) == 4:
            self._first_date = int(first_date)
        else:
            raise ValueError(
                f"First date {first_date} is not a valid 4 digit integer representing a UWY."
            )

        last_date = config["last date"]
        last_date = datetime.strptime(last_date, "%B %Y").strftime("%Y%m")
        last_date_quarter = (
            int(last_date[-2:]) // 3 * 3
        )  # convert to quarter (3,6,9,12)
        self._last_date = int(
            last_date[:-2] + str(last_date_quarter).zfill(2)
        )  # convert to YYYYMM format

        self._segment = config.get("segment", None)
        if self._segment is None:
            raise ValueError(
                "Segment is not specified in the config file. Please specify the segment in the config file."
            )

    def _init_sessions(self, config):
        session_config = config.get("session", {})
        session_path = session_config.get("path")
        if session_path:
            self._session_path = Path(session_path)
        else:
            safe_segment = re.sub(r"[^A-Za-z0-9_\-]", "_", self._segment)
            self._session_path = Path(self._SESSIONS) / f"{safe_segment}.yml"
            self._ensure_session_path_in_config()

    def _ensure_session_path_in_config(self):
        if not self._config_path:
            return
        config = self._config.copy()
        config.setdefault("session", {})
        config["session"]["path"] = str(self._session_path)
        with self._config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def get_segment(self):
        return self._segment

    def get_first_UWY(self):
        return self._first_date

    def get_latest_period(self):
        return self._last_date

    def get_session_path(self) -> Path:
        return self._session_path

    def load_session(self) -> dict:
        session_path = self.get_session_path()
        if not session_path.exists():
            return {}
        with session_path.open("r") as f:
            return yaml.safe_load(f) or {}

    def save_session(self, data: dict) -> None:
        session_path = self.get_session_path()
        session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = data.copy()
        payload.setdefault("segment", self._segment)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with session_path.open("w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
