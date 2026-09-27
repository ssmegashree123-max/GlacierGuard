"""
GlacierGuard Risk Engine
------------------------
Combines ML probabilities, satellite prior, time-to-critical estimate,
and system state into a decision-ready early-warning result.
"""


def classify_risk(p_gradual, p_sudden, lead_time_min=None,
                  satellite_alert=False, system_state="NORMAL"):
    """Return risk level, lead time and alert decision."""

    p_gradual = float(max(0.0, min(1.0, p_gradual)))
    p_sudden = float(max(0.0, min(1.0, p_sudden)))

    if satellite_alert or p_sudden >= 0.80 or system_state in {
        "BREACH", "FLOOD", "DEBRIS_FLOW"
    }:
        level = "RED"
        alert = True
    elif p_gradual >= 0.70:
        level = "ORANGE"
        alert = True
    elif p_gradual >= 0.40:
        level = "YELLOW"
        alert = False
    else:
        level = "GREEN"
        alert = False

    if lead_time_min is not None:
        try:
            lead_time_min = max(0, round(float(lead_time_min), 1))
        except (TypeError, ValueError):
            lead_time_min = None

    return {
        "risk_level": level,
        "alert": alert,
        "lead_time_min": lead_time_min,
        "decision": "ISSUE WARNING" if alert else "MONITOR",
    }
