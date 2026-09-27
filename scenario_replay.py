import math
from datetime import datetime, timedelta, timezone


class ScenarioReplay:
    """
    Escalating synthetic scenario for GlacierGuard demo.

    Provides:
      - warmup_rows()
      - next_row()
      - describe()
      - warmup_end
    """

    def __init__(self, warmup_end=30, total_rows=90):
        self.warmup_end = warmup_end
        self.total_rows = total_rows
        self.index = 0

        self.start_time = datetime.now(timezone.utc)

        # Normal baseline values
        self.base_water = 2.50
        self.base_vibration = 0.04
        self.base_tilt = 0.0
        self.base_rainfall = 2.0
        self.base_turbidity = 5.0
        self.base_temp = -1.5
        self.base_seismic = 0.01

    def _make_row(self, i):
        """
        Generate one telemetry row.

        First warm-up period is stable.
        After warm-up, the hazard gradually escalates.
        """

        timestamp = self.start_time + timedelta(minutes=10 * i)

        if i < self.warmup_end:
            progress = 0.0
        else:
            progress = min(
                1.0,
                (i - self.warmup_end) /
                max(1, self.total_rows - self.warmup_end - 1)
            )

        # Gradual escalation
        water = self.base_water + 0.9 * progress
        vibration = self.base_vibration + 0.14 * progress
        tilt = self.base_tilt + 0.018 * progress
        rainfall = self.base_rainfall + 35.0 * progress
        turbidity = self.base_turbidity + 45.0 * progress
        temperature = self.base_temp + 1.0 * progress
        seismic = self.base_seismic + 0.10 * progress

        return {
            "timestamp_utc": timestamp.isoformat(),

            "water_level_m": round(water, 5),
            "vibration_rms_g": round(vibration, 5),
            "tilt_deg": round(tilt, 5),
            "rainfall_mm_hr": round(rainfall, 3),
            "turbidity_ntu": round(turbidity, 3),
            "ambient_temp_c": round(temperature, 3),
            "seismic_energy": round(seismic, 5),

            "water_valid": 1,
            "vibration_valid": 1,
            "tilt_valid": 1,
            "rainfall_valid": 1,
            "turbidity_valid": 1,

            "sensor_status_water": "synthetic",
            "is_synthetic_water": True,
        }

    def warmup_rows(self):
        """Return stable history needed for rolling features."""
        return [
            self._make_row(i)
            for i in range(self.warmup_end)
        ]

    def next_row(self):
        """Return the next escalating telemetry row."""

        if self.index >= self.total_rows - self.warmup_end:
            return None

        actual_index = self.warmup_end + self.index
        self.index += 1

        return self._make_row(actual_index)

    def describe(self):
        return (
            "Escalating synthetic glacial-hazard scenario: "
            "stable warm-up followed by increasing water level, "
            "vibration, tilt, rainfall, turbidity and seismic activity."
        )