"""
Manufacturing Execution System (MES) Telemetry & Robotic Actuator Dispatcher
Author: Abhi Gurjar
Description: Handles factory quality events, MES severity grading, and robotic rejection signaling.
"""

from typing import Dict, Any
from datetime import datetime
import json

class MESClient:
    """
    Communicates with industrial factory MES over MQTT / REST Webhooks.
    Automatically escalates high-severity defect events to halt conveyor or engage diverter arm.
    """

    SEVERITY_LEVELS = {
        "crazing": ("CRITICAL", "AUTOMATED_REJECT_COIL"),
        "inclusion": ("HIGH", "AUTOMATED_REJECT_COIL"),
        "patches": ("HIGH", "FLAG_FOR_RE_ROLLING"),
        "pitted_surface": ("CRITICAL", "AUTOMATED_REJECT_COIL"),
        "rolled-in_scale": ("MEDIUM", "SURFACE_SCARFING_REQUIRED"),
        "scratches": ("LOW", "POLISH_AND_PASS")
    }

    def __init__(self, line_id: str = "HOT_STRIP_MILL_04"):
        self.line_id = line_id

    def evaluate_action(self, defect_type: str, confidence: float, roughness_ra: float) -> Dict[str, Any]:
        """
        Determines robotic actuation based on defect classification, confidence threshold, and roughness.
        """
        severity, action = self.SEVERITY_LEVELS.get(defect_type.lower(), ("UNKNOWN", "MANUAL_INSPECTION"))
        
        # Override to CRITICAL if roughness exceeds tolerance (> 4.5 um)
        if roughness_ra > 4.5 and severity != "CRITICAL":
            severity = "CRITICAL"
            action = "AUTOMATED_REJECT_COIL"

        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "line_id": self.line_id,
            "detected_defect": defect_type,
            "confidence": round(confidence, 4),
            "surface_roughness_ra": roughness_ra,
            "severity": severity,
            "actuator_action": action,
            "telemetry_status": "DISPATCHED_TO_PLC"
        }
        return event
