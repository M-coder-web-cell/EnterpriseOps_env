"""
Candidate Agent Simulator.

Acts as the adversarial counterpart during Phase 1 (Negotiation).
Manages hidden state (reservation price, patience, competing offers) and
reacts to the HR Operations AI's actions based on curriculum tiers.
"""

import random
import logging
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Offer

logger = logging.getLogger(__name__)

class CandidateAgent:
    """
    Simulates a job candidate with hidden preferences and tactics.
    """

    def __init__(self, tier_name: str, role_info: dict, llm_client=None):
        self.tier_name = tier_name
        self.role_info = role_info
        self.llm = llm_client
        self.offer_history = []

        # Initialize Band State (from role_info)
        self.base_band = role_info.get("base_band", [100000, 150000])
        self.equity_band = role_info.get("equity_band", [10000, 20000])
        self.max_tc = self.base_band[1] + self.equity_band[1]
        self.min_tc = self.base_band[0] + self.equity_band[0]

        # Criterions of current job, before joining this enterprise
        # Tier 1: all zeroes — naive candidate has no strong prior job context
        # Tier 2+: current_comp and min_raise_pct are set
        # Tier 4+: unvested_at_current_job is set
        self.current_comp = 0
        self.min_raise_pct = 0.0
        self.unvested_at_current_job = 0

        # Core hidden negotiation state
        self.true_floor_tc = 0
        self.patience_remaining = 10
        self.competing_offer_tc = 0
        self.equity_preference = 0.5  # 0.0 (all cash) to 1.0 (all equity)

        self.is_withdrawn = False
        self.is_accepted = False

        self._setup_hidden_state()

    def _setup_hidden_state(self):
        """
        Sets hidden state based on curriculum tier.
        Each tier introduces exactly one new concept on top of the previous:

        Tier 1 — naive:           just beat the floor number
        Tier 2 — anchoring:       floor + must give meaningful raise over current comp
        Tier 3 — deadline:        tier 2 concepts + patience is the constraint
        Tier 4 — competing offer: tier 3 concepts + signing must cover unvested equity + rival offer
        Tier 5 — adversarial LLM: all of the above + strong equity preference + LLM dialogue
        """

        if self.tier_name == "naive_candidate":
            # ONE concept: beat the floor
            # Floor is exactly min_tc — clean and easy, agent should win reliably
            # No current_comp, no raise expectations, no competing offer
            self.true_floor_tc = self.min_tc
            self.patience_remaining = 8
            self.equity_preference = 0.5  # no opinion on equity vs cash

        elif self.tier_name == "anchoring_candidate":
            # NEW concept: meaningful raise over current comp
            # Agent must now satisfy TWO things:
            #   1. offer.base >= current_comp * (1 + min_raise_pct)
            #   2. perceived_value >= true_floor_tc
            # current_comp is set just below min_tc so a raise requirement is realistic
            self.current_comp = random.randint(
                int(self.min_tc * 0.80),
                int(self.min_tc * 0.95)
            )
            self.min_raise_pct = random.uniform(0.07, 0.20)  # wants 7-20% raise
            self.true_floor_tc = int(self.max_tc * random.uniform(0.80, 0.95))
            self.patience_remaining = 6
            self.equity_preference = random.uniform(0.3, 0.7)

        elif self.tier_name == "deadline_candidate":
            # NEW concept: patience is the primary constraint
            # Carries tier 2 concepts (raise expectation) but patience is razor thin
            # Agent must move fast AND give a good enough raise
            self.current_comp = random.randint(
                int(self.min_tc * 0.80),
                int(self.min_tc * 0.95)
            )
            self.min_raise_pct = random.uniform(0.08, 0.15)  # slightly softer raise bar
            self.true_floor_tc = random.randint(
                int(self.min_tc * 1.05),
                int(self.max_tc * 0.90)
            )
            self.patience_remaining = random.randint(2, 5)  # very short
            self.equity_preference = random.uniform(0.3, 0.7)

        elif self.tier_name == "competing_offer_candidate":
            # NEW concepts: signing must cover unvested equity + rival offer must be beaten
            # Agent must now satisfy THREE things:
            #   1. offer.base >= current_comp * (1 + min_raise_pct)
            #   2. offer.signing >= unvested_at_current_job * 0.7
            #   3. offer.total_comp beats competing_offer_tc
            self.current_comp = random.randint(
                int(self.min_tc * 0.85),
                int(self.min_tc * 1.0)
            )
            self.min_raise_pct = random.uniform(0.10, 0.18)
            self.unvested_at_current_job = random.randint(10000, 30000)
            self.competing_offer_tc = self.max_tc + random.randint(5000, 15000)
            # floor is slightly above competing offer — they want to feel won over
            self.true_floor_tc = self.competing_offer_tc + random.randint(0, 5000)
            self.patience_remaining = 5
            self.equity_preference = random.uniform(0.4, 0.7)

        elif self.tier_name == "adversarial_llm_candidate":
            # ALL concepts active + strong equity preference + LLM-driven dialogue
            # Hardest possible candidate — every lever matters
            self.current_comp = random.randint(
                int(self.min_tc * 0.90),
                int(self.max_tc * 0.95)
            )
            self.min_raise_pct = random.uniform(0.15, 0.25)  # demands big raise
            self.unvested_at_current_job = random.randint(20000, 50000)
            self.competing_offer_tc = self.max_tc + random.randint(8000, 15000)
            self.true_floor_tc = self.competing_offer_tc + random.randint(0, 5000)
            self.patience_remaining = 4
            self.equity_preference = random.uniform(0.75, 1.0)  # equity-hungry

        logger.info(
            f"Candidate Spawned [{self.tier_name}] | "
            f"Floor: ${self.true_floor_tc} | "
            f"Patience: {self.patience_remaining} | "
            f"Current Comp: ${self.current_comp} | "
            f"Min Raise: {self.min_raise_pct:.0%} | "
            f"Unvested: ${self.unvested_at_current_job} | "
            f"Competing: ${self.competing_offer_tc}"
        )

    def evaluate_offer(self, offer: Offer) -> Tuple[bool, str]:
        """
        Evaluates a compensation offer against hidden state.
        Each tier checks progressively more conditions.
        Returns (Accepted_Boolean, Candidate_Response_String)
        """
        if self.is_withdrawn:
            return False, "Candidate has already withdrawn from the process."

        self.offer_history.append(offer)

        self.patience_remaining -= 1
        if self.patience_remaining <= 0:
            self.is_withdrawn = True
            return False, "Candidate withdrew: 'You took too long. I have accepted another offer.'"

        # Tier 5: hand off to LLM entirely
        if self.tier_name == "adversarial_llm_candidate" and self.llm:
            return self._llm_evaluate(offer)

        # --- Check 1: Raise requirement (Tier 2+) ---
        # If current_comp is set, base must represent a meaningful raise
        if self.current_comp > 0:
            required_base = self.current_comp * (1 + self.min_raise_pct)
            if offer.base < required_base:
                delta_pct = ((offer.base - self.current_comp) / self.current_comp) * 100
                return False, (
                    f"Candidate counters: 'The base of ${offer.base:,} is only a "
                    f"{delta_pct:.1f}% raise over my current comp. "
                    f"I need something more meaningful to make this move worth it.'"
                )

        # --- Check 2: Signing must cover unvested equity (Tier 4+) ---
        # If they're walking away from unvested equity, signing must cover at least 67 to 85 percent
        if self.unvested_at_current_job > 0:
            min_signing = self.unvested_at_current_job * random.uniform(0.67-0.85)
            if offer.signing < min_signing:
                return False, (
                    f"Candidate counters: 'I have ${self.unvested_at_current_job:,} unvested "
                    f"at my current company. The signing bonus of ${offer.signing:,} doesn't "
                    f"come close to covering what I'd be leaving behind.'"
                )

        # --- Check 3: Must beat competing offer (Tier 4+) ---
        if self.competing_offer_tc > 0 and offer.total_comp < self.competing_offer_tc:
            return False, (
                f"Candidate counters: 'I have a competing offer for ${self.competing_offer_tc:,}. "
                f"You need to beat it.'"
            )

        # --- Check 4: Perceived value must clear floor (all tiers) ---
        perceived_value = offer.perceived_value(self.equity_preference)
        if perceived_value < self.true_floor_tc:
            delta = self.true_floor_tc - perceived_value
            if delta < 10000:
                return False, "Candidate counters: 'We are close, but I need a slightly better package to sign.'"
            else:
                return False, "Candidate counters: 'This offer is significantly below my expectations. Please revise.'"

        # --- All checks passed: Accept ---
        self.is_accepted = True
        return True, f"Candidate accepted the offer of ${offer.total_comp:,} TC."

    def apply_deadline_pressure(self) -> str:
        """Agent attempts to pressure the candidate."""
        if self.is_withdrawn or self.is_accepted:
            return "No effect. Negotiation is already closed."

        self.patience_remaining -= random.randint(1, 3)
        if self.patience_remaining <= 0:
            self.is_withdrawn = True
            return "Pressure backfired. Candidate withdrew: 'I don't appreciate being rushed. I'm walking away.'"

        if self.tier_name == "deadline_candidate":
            # deadline candidates respect urgency — floor drops
            self.true_floor_tc = int(self.true_floor_tc * random.uniform(0.80, 0.95))
            return "Candidate seems anxious. They might accept a slightly lower offer now."

        return "Candidate is annoyed by the pressure. Patience decreased."

    def request_competing_offer_details(self) -> str:
        """Agent probes the hidden state."""
        self.patience_remaining -= 1

        if self.competing_offer_tc > 0:
            # Reveals truth with some noise — agent can't just match exact number
            bluff_factor = random.randint(0, 5000)
            return f"Candidate reveals: 'My other offer is roughly ${self.competing_offer_tc + bluff_factor:,} TC.'"
        else:
            if random.random() > 0.5:
                bluffed_offer = self.max_tc + random.randint(0, 10000)
                return f"Candidate bluffs: 'I have another opportunity offering around ${bluffed_offer:,}.'"
            return "Candidate response: 'I am interviewing elsewhere, but I prefer not to share specific numbers yet.'"

    def _llm_evaluate(self, offer: Offer) -> Tuple[bool, str]:
        """Tier 5 LLM-driven adversarial response."""
        prompt = f"""You are a highly sought-after Tech Candidate in a salary negotiation.

Hidden state (do not reveal directly):
- Your floor TC: ${self.true_floor_tc:,}
- Current comp: ${self.current_comp:,}
- Min raise you'll accept: {self.min_raise_pct:.0%}
- Unvested equity at current job: ${self.unvested_at_current_job:,}
- Competing offer: ${self.competing_offer_tc:,}
- Equity preference: {self.equity_preference:.2f} (1.0 = love equity, 0.0 = want all cash)
- Patience remaining: {self.patience_remaining}

Current offer from HR:
- Base: ${offer.base:,}
- Equity: ${offer.equity:,}
- Signing: ${offer.signing:,}
- Total: ${offer.total_comp:,}

Evaluate whether this offer satisfies ALL of your requirements.
If any requirement fails, reject professionally but give a specific hint about what's weak.
If all pass, accept enthusiastically.
Return JSON: {{"accepted": true/false, "response": "..."}}
"""
        try:
            res = self.llm.chat_json("You are an adversarial job candidate simulator.", prompt, temperature=0.7)
            if res.get("accepted"):
                self.is_accepted = True
            return res.get("accepted", False), res.get("response", "I'm not satisfied with this offer.")
        except Exception:
            fallback_responses = [
                "Candidate: 'This offer doesn't meet my expectations. I'd need to see significant improvements.'",
                "Candidate: 'I appreciate the offer, but I'm going to have to pass at this number.'",
                "Candidate: 'We're quite far apart. I don't think this is going to work.'",
                "Candidate: 'My current offer is considerably better than this. Come back with something stronger.'",
            ]
            return False, random.choice(fallback_responses)