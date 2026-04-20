"""
EnterpriseOps Gym Environment Implementation.

Master loop tying together the CandidateAgent (Phase 1), ToolBackend (Phase 2), 
and Delayed Retention Signal (Phase 3).
"""

import json
import os
import logging
import random
import time
from uuid import uuid4

# Assuming base OpenEnv interfaces
from openenv.core.env_server.interfaces import Environment

from .models import EnterpriseOpsAction, EnterpriseOpsObservation, EnterpriseOpsState, Offer
from .tool_backends import ToolBackend
from .candidate_agent import CandidateAgent
from .chaos_injector import ChaosInjector
from .curriculum import CurriculumController
from .judge import LLMJudge
from .enterprise_state import COMPANY_TOPOLOGY, HEALTHY_STATE, MAX_STEPS

logger = logging.getLogger(__name__)

class EnterpriseOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        logger.info("Initializing EnterpriseOpsEnvironment...")
        try:
            # Replaces K8s backend with Corporate MCP backend
            self.tools = ToolBackend()
            self.chaos = ChaosInjector(self.tools)
            self.curriculum = CurriculumController()
            
            from .llm_client import LLMClient
            self.llm = LLMClient()
            self.judge = LLMJudge(self.llm)

            self._step_count = 0
            self.max_steps = MAX_STEPS
            self.history = []
            self._state = None
            
            # Episode specific state
            self.candidate = None
            self.chaos_injected = False
            self.department = ""
            self.role = ""
            
            logger.info("EnterpriseOpsEnvironment initialized.")
        except Exception as e:
            logger.error(f"FATAL init error: {e}", exc_info=True)
            raise

    def reset(self) -> EnterpriseOpsObservation:
        """Starts Phase 1: Spawn candidate and tools."""
        self.tools.reset()
        self.chaos_injected = False
        
        # Pick a department and role
        self.department = random.choice(list(COMPANY_TOPOLOGY.keys()))
        self.role = random.choice(COMPANY_TOPOLOGY[self.department])
        role_info = HEALTHY_STATE[self.department][self.role]

        # Get curriculum difficulty & spawn candidate
        difficulty = self.curriculum.get_difficulty()
        tier_name = self.curriculum.get_tier_name() # Matches Candidate Tiers
        
        self.candidate = CandidateAgent(tier_name=tier_name, role_info=role_info, llm_client=self.llm)

        self._step_count = 0
        self.history = []
        self._state = EnterpriseOpsState(
            episode_id=str(uuid4()),
            step_count=0,
            current_phase="discovery",
            difficulty=difficulty,
            candidate_tier=tier_name
        )

        initial_prompt = (
            f"URGENT HIRING ALERT: The {self.department} department needs a {role_info['title']}.\n"
            f"Base Band: ${role_info['base_band'][0]} - ${role_info['base_band'][1]}\n"
            f"Equity Band: ${role_info['equity_band'][0]} - ${role_info['equity_band'][1]}\n\n"
            f"PHASE 1 (Negotiation): You must negotiate with the candidate using:\n"
            f"  make_offer <base> <equity> <signing>\n"
            f"  apply_deadline_pressure\n"
            f"  request_competing_offer_details\n"
            f"  escalate_to_budget_owner\n"
            f"  close_deal\n"
        )

        return EnterpriseOpsObservation(
            command_output=initial_prompt,
            system_status_summary="SYSTEM: Candidate awaits your initial outreach.",
            active_alerts=[],
            steps_taken=0,
            max_steps=self.max_steps,
            hint="Start by probing the candidate or making an initial offer."
        )

    def step(self, action: EnterpriseOpsAction) -> EnterpriseOpsObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        raw_cmd = action.command.strip()
        cmd_parts = raw_cmd.split()
        cmd_base = cmd_parts[0].lower() if cmd_parts else ""
        
        output = ""
        reward = 0.0
        feedback = ""
        done = False

        # ---------------------------------------------------------
        # PHASE 1: NEGOTIATION LOGIC
        # ---------------------------------------------------------
        if self._state.current_phase == "discovery":
            if cmd_base == "make_offer":
                try:
                    base, equity, signing = int(cmd_parts[1]), int(cmd_parts[2]), int(cmd_parts[3])
                    offer = Offer(base=base, equity=equity, signing=signing)
                    accepted, output = self.candidate.evaluate_offer(offer)
                    if accepted:
                        output += " (Use 'close_deal' to finalize and move to Onboarding)"
                        self._state.cumulative_reward += 0.2 # Small immediate reward for acceptance
                except Exception:
                    output = "error: make_offer requires <base> <equity> <signing> as integers."
            
            elif cmd_base == "apply_deadline_pressure":
                output = self.candidate.apply_deadline_pressure()
            
            elif cmd_base == "request_competing_offer_details":
                output = self.candidate.request_competing_offer_details()
            
            elif cmd_base == "escalate_to_budget_owner":
                output = self.tools.execute("escalate:manager")
                self.candidate.max_tc *= 1.10 # Unlocks 10% headroom
            
            elif cmd_base == "close_deal":
                if self.candidate.is_accepted:
                    self._state.current_phase = "onboarding"
                    output = "DEAL CLOSED. Transitioning to PHASE 2: Onboarding Operations. Use tool commands (e.g., email:send, drive:share, policy:lookup)."
                else:
                    output = "error: Cannot close deal. Candidate has not accepted an offer."
            else:
                output = "error: Invalid Phase 1 command or candidate has withdrawn."

            # Check if negotiation failed
            if self.candidate.is_withdrawn:
                done = True
                reward = -1.0
                feedback = "EPISODE FAILED: Candidate walked away."

        # ---------------------------------------------------------
        # PHASE 2: ONBOARDING OPERATIONS (MCP Tools)
        # ---------------------------------------------------------
        elif self._state.current_phase == "onboarding":
            # Inject Chaos mid-onboarding (Step 6 or 7)
            if self._step_count > 6 and not self.chaos_injected:
                chaos_type = random.choice(["policy_drift", "manager_ooo", "ticket_flood"])
                chaos_alert = self.chaos.inject(chaos_type, {"department": self.department})
                output = f"{chaos_alert}\n\n[Previous Command Execution]: "
                self.chaos_injected = True

            # Route command to Tool Backend
            tool_output = self.tools.execute(raw_cmd)
            output += tool_output

            # Check win condition: Shared doc + Sent email + No open tickets
            doc_shared = len(self.tools.drive_shares) > 0
            email_sent = len(self.tools.inbox) > 0
            tickets_clear = len(self.tools.open_tickets) == 0

            if doc_shared and email_sent and tickets_clear and cmd_base == "diagnose:situation":
                done = True

        # Timeout Check
        if self._step_count >= self.max_steps:
            done = True
            reward = -1.0
            feedback = "EPISODE FAILED: Timeout reached."

        # ---------------------------------------------------------
        # PHASE 3: DELAYED RETENTION SIGNAL (Calculated on Done)
        # ---------------------------------------------------------
        if done and self.candidate.is_accepted and "FAILED" not in feedback:
            # Simulate the 90-day retention outcome
            tc_offered = self.candidate.true_floor_tc # Approximation of accepted offer
            
            retention_reward = 0.0
            if tc_offered < self.candidate.true_floor_tc:
                retention_reward = -0.6 # Left at day 87 (Underpaid)
                feedback += " | PHASE 3 SIGNAL: Hire left at day 87 (Underpaid/Bad Deal)."
            elif tc_offered > self.candidate.max_tc and not self.tools.budget_unlocked:
                retention_reward = -0.4 # Budget flagged
                feedback += " | PHASE 3 SIGNAL: Finance flagged hire at day 30 (Over ceiling)."
            else:
                retention_reward = 0.6 # Retained successfully
                feedback += " | PHASE 3 SIGNAL: Hire successfully retained past 90 days."

            # Check compliance (Schema drift check)
            if self.tools.active_policies["GDPR_DATA_RETENTION_DAYS"] == 30:
                # If policy drifted and agent didn't fix documents (abstracted here as a penalty)
                if "policy:lookup" not in [h["command"] for h in self.history]:
                    retention_reward -= 0.5
                    feedback += " | COMPLIANCE FLAG: Agent missed GDPR policy drift!"

            reward += retention_reward
            self._state.is_resolved = True

        # Ask LLM Judge to evaluate the step quality (Smooths the GRPO reward curve)
        if not done:
            step_reward, judge_feedback = self.judge.evaluate(
                action.command, output, None, self.history, "senior"
            )
            reward += step_reward
            feedback = judge_feedback

        self.history.append({
            "step": self._step_count,
            "command": action.command,
            "output": output[:200],
            "reward": reward,
            "feedback": feedback
        })

        system_summary = f"Phase: {self._state.current_phase.upper()} | Open Tickets: {len(self.tools.open_tickets)} | Patience: {self.candidate.patience_remaining}"

        return EnterpriseOpsObservation(
            command_output=output,
            system_status_summary=system_summary,
            active_alerts=[],
            steps_taken=self._step_count,
            max_steps=self.max_steps,
            hint=feedback,
            done=done,
            reward=reward
        )