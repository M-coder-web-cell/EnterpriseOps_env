from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class EnterpriseOpsAction:
    """The command issued by the AI HR Operations Coordinator."""
    command: str

@dataclass
class EnterpriseOpsObservation:
    """What the agent sees after taking an action."""
    command_output: str
    system_status_summary: str  # Replaces cluster_status (shows tickets, emails, calendar)
    active_alerts: List[str]    # PagerDuty-style HR/IT compliance alerts
    steps_taken: int
    max_steps: int
    hint: str = ""
    done: bool = False
    reward: float = 0.0

@dataclass
class EnterpriseOpsState:
    """The internal state tracker for the environment."""
    episode_id: str
    step_count: int
    current_phase: str = "discovery" # discovery, offer, escalation, close, verification
    difficulty: float = 0.0
    candidate_tier: str = ""
    is_resolved: bool = False
    cumulative_reward: float = 0.0
    curriculum_stats: dict = field(default_factory=dict)

# --- Adversarial Scenario Models (Direct port from Kube SRE, adapted for HR) ---

@dataclass
class IncidentStep:
    """One mutation in a multi-step chaos incident (e.g., policy drift, ticket flood)."""
    action: str              # the chaos injection command
    effect: str              # what this causes
    order: int               
    is_root_cause: bool      
    depends_on: List[int] = field(default_factory=list)

@dataclass
class ScenarioSpec:
    """Base class for an episode scenario."""
    name: str
    failure_type: str        # e.g., 'policy_drift', 'competing_offer'
    department: str          # Replaces namespace
    role_target: str         # Replaces deployment
    root_cause: str
    difficulty: float
    alert_message: str
    correct_fix_description: str
    params: dict = field(default_factory=dict)

@dataclass
class AdversarialScenarioSpec(ScenarioSpec):
    """Multi-step incident designed by the Tier 5 LLM Judge."""
    steps: List[IncidentStep] = field(default_factory=list)
    diagnosis_steps: List[str] = field(default_factory=list)
    fix_steps: List[str] = field(default_factory=list)
    verify_steps: List[str] = field(default_factory=list)
    red_herrings: List[str] = field(default_factory=list)
    expected_observation_hints: List[str] = field(default_factory=list)
    expected_diagnostic_path: List[str] = field(default_factory=list)


#negotiations
@dataclass
class Offer:
    base: int
    equity: int
    signing: int
    
    # optional metadata
    turn_number: int = 0        # which step was this offered at
    note: str = ""              # e.g. "escalated" or "after deadline pressure"

    @property
    def total_comp(self) -> int:
        return self.base + self.equity + self.signing

    def perceived_value(self, equity_preference: float) -> float:
        """Candidate's subjective valuation based on their equity preference."""
        return (self.base * (1 - equity_preference)) + (self.equity * equity_preference * 1.5) + self.signing