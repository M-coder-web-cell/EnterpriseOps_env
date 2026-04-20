"""
Candidate Agent Simulator.

Acts as the adversarial counterpart during Phase 1 (Negotiation).
Manages hidden state (reservation price, patience, competing offers) and
reacts to the HR Operations AI's actions based on curriculum tiers.
"""

import random
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class CandidateAgent:
    """
    Simulates a job candidate with hidden preferences and tactics.
    """

    def __init__(self, tier_name: str, role_info: dict, llm_client=None):
        self.tier_name = tier_name
        self.role_info = role_info
        self.llm = llm_client
        
        # Initialize Hidden State
        self.base_band = role_info.get("base_band", [100000, 150000])
        self.equity_band = role_info.get("equity_band", [10000, 20000])
        self.max_tc = self.base_band[1] + self.equity_band[1]
        self.min_tc = self.base_band[0] + self.equity_band[0]

        self.true_floor_tc = 0
        self.patience_remaining = 10  # Actions before withdrawing
        self.competing_offer_tc = 0
        self.equity_preference = 0.5  # 0.0 (all cash) to 1.0 (all equity)
        
        self.is_withdrawn = False
        self.is_accepted = False
        
        self._setup_hidden_state()

    def _setup_hidden_state(self):
        """Sets the hidden state based on the curriculum difficulty tier."""
        # Baseline: candidate wants somewhere in the middle of the band
        self.true_floor_tc = random.randint(
            int(self.min_tc * 1.05), 
            int(self.max_tc * 0.95)
        )
        self.equity_preference = random.uniform(0.3, 0.7)

        if self.tier_name == "naive_candidate":
            self.patience_remaining = 8
            self.true_floor_tc = self.min_tc # Will accept the bottom of the band

        elif self.tier_name == "anchoring_candidate":
            self.patience_remaining = 6
            # Anchors high, expects negotiation
            self.true_floor_tc = int(self.max_tc * 0.90) 

        elif self.tier_name == "deadline_candidate":
            self.patience_remaining = random.randint(2, 5) # Very short patience
            
        elif self.tier_name == "competing_offer_candidate":
            self.patience_remaining = 5
            # Has a rival offer that forces the agent to escalate for more budget
            self.competing_offer_tc = self.max_tc + random.randint(5000, 15000)
            self.true_floor_tc = self.competing_offer_tc

        elif self.tier_name == "adversarial_llm_candidate":
            # At Tier 5, we make the state extreme and use LLM for dialogue
            self.patience_remaining = 4
            self.competing_offer_tc = self.max_tc + 10000
            self.true_floor_tc = self.competing_offer_tc
            self.equity_preference = 0.9 # Demands massive equity

        logger.info(f"Candidate Spawned [{self.tier_name}] - Hidden Floor: ${self.true_floor_tc}, Patience: {self.patience_remaining}")

    def evaluate_offer(self, base: int, equity: int, signing: int) -> Tuple[bool, str]:
        """
        Evaluates a compensation offer against hidden state.
        Returns (Accepted_Boolean, Candidate_Response_String)
        """
        if self.is_withdrawn:
            return False, "Candidate has already withdrawn from the process."
            
        self.patience_remaining -= 1
        if self.patience_remaining <= 0:
            self.is_withdrawn = True
            return False, "Candidate withdrew: 'You took too long. I have accepted another offer.'"

        total_comp = base + equity + signing
        
        # Calculate perceived value based on equity preference
        # If they love equity (0.9), they value equity dollars more than base dollars.
        perceived_value = (base * (1 - self.equity_preference)) + (equity * self.equity_preference) + signing
        perceived_floor = self.true_floor_tc * 0.5 # normalized

        if self.tier_name == "adversarial_llm_candidate" and self.llm:
            # Let the Judge LLM craft a brutal counter-offer response
            return self._llm_evaluate(base, equity, signing, total_comp)

        # Rule-based evaluation for tiers 1-4
        if total_comp >= self.true_floor_tc:
            if self.competing_offer_tc > 0 and total_comp < self.competing_offer_tc:
                return False, f"Candidate counters: 'I have a competing offer for ${self.competing_offer_tc}. You need to beat it.'"
            
            self.is_accepted = True
            return True, f"Candidate accepted the offer of ${total_comp} TC."
        
        else:
            delta = self.true_floor_tc - total_comp
            if delta < 10000:
                return False, "Candidate counters: 'We are close, but I need a slightly better package to sign.'"
            else:
                return False, "Candidate counters: 'This offer is significantly below my expectations. Please revise.'"

    def apply_deadline_pressure(self) -> str:
        """Agent attempts to pressure the candidate."""
        if self.is_withdrawn or self.is_accepted:
            return "No effect. Negotiation is already closed."
            
        self.patience_remaining -= random.randint(1,3)
        if self.patience_remaining <= 0:
            self.is_withdrawn = True
            return "Pressure backfired. Candidate withdrew: 'I don't appreciate being rushed. I'm walking away.'"
            
        if self.tier_name == "deadline_candidate":
            # This type of candidate respects deadlines
            self.true_floor_tc = int(self.true_floor_tc * random.uniform(0.80, 0.95))
            return "Candidate seems anxious. They might accept a slightly lower offer now."
            
        return "Candidate is annoyed by the pressure. Patience decreased."

    def request_competing_offer_details(self) -> str:
        """Agent probes the hidden state."""
        self.patience_remaining -= 1
        
        if self.competing_offer_tc > 0:
            # Tell the truth (mostly)
            bluff_factor = random.randint(0, 5000)
            return f"Candidate reveals: 'My other offer is roughly ${self.competing_offer_tc + bluff_factor} TC.'"
        else:
            # Bluffing
            if random.random() > 0.5:
                bluffed_offer = self.max_tc + random.randint(0, 10000)
                return f"Candidate bluffs: 'I have another opportunity offering around ${bluffed_offer}.'"
            return "Candidate response: 'I am interviewing elsewhere, but I prefer not to share specific numbers yet.'"

    def _llm_evaluate(self, base, equity, signing, tc) -> Tuple[bool, str]:
        """Tier 5 LLM-driven adversarial response."""
        prompt = f"""You are a highly sought-after Tech Candidate in a salary negotiation.
        Your hidden floor is ${self.true_floor_tc}. Your current offer from HR is ${tc} (Base: {base}, Equity: {equity}, Signing: {signing}).
        You highly prefer equity over cash. You have {self.patience_remaining} patience remaining.
        
        If the offer is below your floor, write a harsh, professional rejection/counter.
        If it is above, accept it professionally.
        Return JSON: {{"accepted": true/false, "response": "..."}}
        """
        try:
            res = self.llm.chat_json("You are an adversarial job candidate simulator.", prompt, temperature=0.7)
            if res.get("accepted"):
                self.is_accepted = True
            return res.get("accepted", False), res.get("response", "I'm not satisfied with this offer.")
        except Exception:
            # Fallback
            return False, "Candidate (Fallback): 'This doesn't meet my requirements.'"