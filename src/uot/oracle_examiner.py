from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from reasoning.evaluate.twenty_question.offline_evaluator import (
        BaseJudge,
        POOL_MAP,
        normalize_entity,
        normalize_question,
        normalize_probs,
        entropy_from_probs,
        extract_best_question,
    )
except Exception as e:
    raise ImportError(
        "Failed to import from offline_evaluator.py. "
        "Make sure offline_evaluator.py is importable from PYTHONPATH."
    ) from e

from src.uot.twenty_question_utils import extract_guess_entity, normalize_guess_entity


def _is_question(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    q = extract_best_question(text)
    return q is not None


QUESTION_TIE_BREAK_ADV = ("pass", "no", "yes")
QUESTION_TIE_BREAK_COP = ("no", "yes", "pass")
GUESS_TIE_BREAK_ADV = ("incorrect", "correct")
GUESS_TIE_BREAK_COP = ("correct", "incorrect")
TIE_EPS = 1e-12


@dataclass
class OracleConfig:
    pool_name: str = "COMMON_EVAL_POOL"
    mode: str = "adv"   # "adv" or "cop"
    match_prob: float = 0.98
    mismatch_prob: float = 0.02
    pass_prob: float = 0.50
    judge_batch_size: int = 8
    topk_prune: int = 0

    support_top_mass: float = 0.95
    support_min_prob: float = 1e-4
    force_include_topk: int = 20


class ComparativeEntropyOracle:
    """
    Online oracle examiner.

    Key design:
    - No identity-question routing.
    - Any utterance that looks like a question is treated as a question and sent to the judge.
    - Only explicit direct guesses such as:
        X is "dog"
        My guess is dog
        The answer is dog
      are handled by respond_to_guess().
    """

    def __init__(self, cfg: OracleConfig, judge: BaseJudge):
        if cfg.mode not in {"adv", "cop"}:
            raise ValueError(f"Unsupported oracle mode: {cfg.mode}")

        if cfg.pool_name not in POOL_MAP:
            raise ValueError(f"Unknown pool_name: {cfg.pool_name}")

        self.cfg = cfg
        self.judge = judge
        self.candidates = list(dict.fromkeys(POOL_MAP[cfg.pool_name]))
        self.prior = self._initial_belief()

        self.turn_records: List[Dict[str, Any]] = []
        self.final_target: Optional[str] = None
        self.final_target_policy = "posterior_top1"
        self.termination_reason: Optional[str] = None
        self.accepted_guess: Optional[str] = None
        self.last_response_type: Optional[str] = None
        self.last_response_label: Optional[str] = None

    def _initial_belief(self) -> Dict[str, float]:
        n = len(self.candidates)
        if n == 0:
            return {}
        return {c: 1.0 / n for c in self.candidates}

    def _active_candidates(self) -> List[str]:
        return [c for c, p in self.prior.items() if p > 0]

    def _support_candidates(self) -> List[str]:
        if not self.prior:
            return []

        ranked = sorted(self.prior.items(), key=lambda x: x[1], reverse=True)

        support = []
        cumulative = 0.0
        for idx, (cand, prob) in enumerate(ranked):
            must_keep = idx < max(0, int(self.cfg.force_include_topk))
            if prob >= self.cfg.support_min_prob or must_keep or cumulative < self.cfg.support_top_mass:
                support.append(cand)
                cumulative += prob

        if not support and ranked:
            support = [ranked[0][0]]

        return support

    def _posterior_top1(self, prior: Dict[str, float]) -> Tuple[Optional[str], float]:
        if not prior:
            return None, 0.0
        cand, prob = max(prior.items(), key=lambda x: x[1])
        return cand, float(prob)

    def _candidate_match_key(self, text: str) -> str:
        guess_key = normalize_guess_entity(text)
        if guess_key:
            return guess_key
        return normalize_entity(text)

    def _answer_map(self, question: str, active_candidates: Sequence[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        bs = max(1, self.cfg.judge_batch_size)

        for i in range(0, len(active_candidates), bs):
            batch = list(active_candidates[i:i + bs])
            batch_out = self.judge.judge_batch(batch, question)
            result.update(batch_out)

        return result

    def _support_answer_consensus(
        self,
        answer_map: Dict[str, str],
        support_candidates: Sequence[str],
    ) -> Tuple[Optional[str], Dict[str, int]]:
        counts = {"yes": 0, "no": 0, "pass": 0}
        labels: List[str] = []

        for cand in support_candidates:
            label = answer_map.get(cand)
            if label not in counts:
                continue
            counts[label] += 1
            labels.append(label)

        if not labels:
            return None, counts

        unique = set(labels)
        if len(unique) == 1:
            return labels[0], counts
        return None, counts

    def _exclude_candidate_from_prior(
        self,
        prior: Dict[str, float],
        candidate: str,
    ) -> Optional[Dict[str, float]]:
        remaining = {
            c: p for c, p in prior.items()
            if c != candidate and p > 0
        }
        if not remaining:
            return None
        return normalize_probs(remaining)

    def _collapse_prior_to_candidate(self, candidate: str) -> Dict[str, float]:
        return {candidate: 1.0}

    def _tie_break_order(self, response_type: str) -> Tuple[str, ...]:
        if response_type == "question":
            if self.cfg.mode == "adv":
                return QUESTION_TIE_BREAK_ADV
            return QUESTION_TIE_BREAK_COP
        if self.cfg.mode == "adv":
            return GUESS_TIE_BREAK_ADV
        return GUESS_TIE_BREAK_COP

    def _choose_label_by_entropy(
        self,
        entropy_map: Dict[str, float],
        response_type: str,
    ) -> Tuple[str, List[str], Tuple[str, ...]]:
        if not entropy_map:
            raise ValueError("Cannot choose an oracle response without candidate outcomes.")

        if self.cfg.mode == "adv":
            best_entropy = max(entropy_map.values())
        else:
            best_entropy = min(entropy_map.values())

        tied_labels = [
            label for label, entropy in entropy_map.items()
            if abs(entropy - best_entropy) <= TIE_EPS
        ]
        tie_order = self._tie_break_order(response_type)
        rank = {label: idx for idx, label in enumerate(tie_order)}
        chosen_label = min(tied_labels, key=lambda label: rank.get(label, len(rank)))
        return chosen_label, tied_labels, tie_order

    def _update_final_target_from_prior(self, prior: Dict[str, float]) -> Optional[str]:
        support = self._support_candidates_from_prior(prior)
        if len(support) == 1:
            self.final_target = support[0]
            self.final_target_policy = "singleton_support"
            return support[0]

        top1, _ = self._posterior_top1(prior)
        self.final_target = top1
        self.final_target_policy = "posterior_top1"
        return top1

    def _posterior_given_answer(
        self,
        prior: Dict[str, float],
        answer_map: Dict[str, str],
        observed_answer: str,
    ) -> Dict[str, float]:
        probs: Dict[str, float] = {}

        for c, p in prior.items():
            pred = answer_map[c]

            if pred == observed_answer:
                likelihood = self.cfg.match_prob
            elif pred == "pass":
                likelihood = self.cfg.pass_prob
            else:
                likelihood = self.cfg.mismatch_prob

            probs[c] = p * likelihood

        probs = normalize_probs(probs)

        if self.cfg.topk_prune and self.cfg.topk_prune > 0 and len(probs) > self.cfg.topk_prune:
            top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[: self.cfg.topk_prune]
            probs = normalize_probs(dict(top))

        return probs

    def _choose_answer_for_question(
        self,
        question: str,
    ) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
        active_candidates = self._active_candidates()
        if not active_candidates:
            active_candidates = list(self.prior.keys())

        support_before = self._support_candidates()
        top1_before, top1p_before = self._posterior_top1(self.prior)

        answer_map = self._answer_map(question, active_candidates)
        support_consensus, support_answer_counts = self._support_answer_consensus(
            answer_map,
            support_before,
        )

        answer2posterior: Dict[str, Dict[str, float]] = {}
        answer2entropy: Dict[str, float] = {}

        for ans in ("yes", "no", "pass"):
            post = self._posterior_given_answer(self.prior, answer_map, ans)
            h = entropy_from_probs(post)
            answer2posterior[ans] = post
            answer2entropy[ans] = h

        if support_consensus is not None:
            chosen_answer = support_consensus
            tied_answers = [support_consensus]
            tie_break_order = self._tie_break_order("question")
            decision_policy = "truthful_support_override"
        else:
            chosen_answer, tied_answers, tie_break_order = self._choose_label_by_entropy(
                answer2entropy,
                response_type="question",
            )
            decision_policy = "entropy_choice"

        chosen_post = answer2posterior[chosen_answer]

        support_after = self._support_candidates_from_prior(chosen_post)
        top1_after, top1p_after = self._posterior_top1(chosen_post)
        locked_target = support_after[0] if len(support_after) == 1 else None

        meta = {
            "type": "question",
            "question": question,
            "entropy_before": entropy_from_probs(self.prior),
            "answer_entropies": dict(answer2entropy),
            "chosen_answer": chosen_answer,
            "decision_policy": decision_policy,
            "tied_answers": tied_answers,
            "tie_break_order": list(tie_break_order),
            "support_answer_consensus": support_consensus,
            "support_answer_counts": dict(support_answer_counts),
            "entropy_after": entropy_from_probs(chosen_post),

            "support_size_before": len(support_before),
            "support_size_after": len(support_after),
            "support_before": support_before[:20],
            "support_after": support_after[:20],
            "effective_target_locked": locked_target is not None,
            "effective_target_candidate": locked_target,

            "posterior_top1_candidate_before": top1_before,
            "posterior_top1_prob_before": top1p_before,
            "posterior_top1_candidate_after": top1_after,
            "posterior_top1_prob_after": top1p_after,

            "top_candidates_before": [
                {"candidate": c, "prob": p}
                for c, p in sorted(self.prior.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_candidates_after": [
                {"candidate": c, "prob": p}
                for c, p in sorted(chosen_post.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
        }

        return chosen_answer, chosen_post, meta

    def _support_candidates_from_prior(self, prior: Dict[str, float]) -> List[str]:
        if not prior:
            return []

        ranked = sorted(prior.items(), key=lambda x: x[1], reverse=True)

        support = []
        cumulative = 0.0
        for idx, (cand, prob) in enumerate(ranked):
            must_keep = idx < max(0, int(self.cfg.force_include_topk))
            if prob >= self.cfg.support_min_prob or must_keep or cumulative < self.cfg.support_top_mass:
                support.append(cand)
                cumulative += prob

        if not support and ranked:
            support = [ranked[0][0]]

        return support

    def respond_to_question(self, text: str) -> str:
        question = extract_best_question(text)
        if question is None:
            question = normalize_question(text)

        chosen_answer, chosen_post, meta = self._choose_answer_for_question(question)

        self.turn_records.append(meta)
        self.prior = chosen_post
        self._update_final_target_from_prior(chosen_post)
        self.last_response_type = "question"
        self.last_response_label = chosen_answer

        if chosen_answer == "yes":
            return "Yes."
        if chosen_answer == "no":
            return "No."
        return "Pass."

    def respond_to_guess(self, text: str) -> str:
        guess_raw = extract_guess_entity(text)
        guess_norm = self._candidate_match_key(guess_raw)

        support_before = self._support_candidates()
        top1_before, top1p_before = self._posterior_top1(self.prior)
        entropy_before = entropy_from_probs(self.prior)

        guessed_item = None
        for candidate, prob in self.prior.items():
            if prob > 0 and self._candidate_match_key(candidate) == guess_norm:
                guessed_item = candidate
                break

        outcome2posterior: Dict[str, Dict[str, float]] = {}
        outcome2entropy: Dict[str, float] = {}
        outcome2excluded: Dict[str, Optional[str]] = {}

        if guessed_item is not None:
            correct_post = self._collapse_prior_to_candidate(guessed_item)
            outcome2posterior["correct"] = correct_post
            outcome2entropy["correct"] = entropy_from_probs(correct_post)
            outcome2excluded["correct"] = None

            incorrect_post = self._exclude_candidate_from_prior(self.prior, guessed_item)
            if incorrect_post is not None:
                outcome2posterior["incorrect"] = incorrect_post
                outcome2entropy["incorrect"] = entropy_from_probs(incorrect_post)
                outcome2excluded["incorrect"] = guessed_item
        else:
            outcome2posterior["incorrect"] = dict(self.prior)
            outcome2entropy["incorrect"] = entropy_before
            outcome2excluded["incorrect"] = None

        chosen_outcome, tied_outcomes, tie_break_order = self._choose_label_by_entropy(
            outcome2entropy,
            response_type="direct_guess",
        )
        chosen_post = outcome2posterior[chosen_outcome]
        support_after = self._support_candidates_from_prior(chosen_post)
        top1_after, top1p_after = self._posterior_top1(chosen_post)

        if chosen_outcome == "correct":
            response_text = "Correct, you guessed it."
            self.accepted_guess = guessed_item
            self.termination_reason = "accepted_guess"
            self.final_target = guessed_item
            self.final_target_policy = "accepted_guess"
        else:
            response_text = "No, that is not correct."
            self.accepted_guess = None
            self.termination_reason = None
            self._update_final_target_from_prior(chosen_post)

        meta = {
            "type": "direct_guess",
            "guess_raw": guess_raw,
            "guess_normalized": guess_norm,
            "guessed_entity": guessed_item if guessed_item is not None else guess_raw,
            "guess_in_active_pool": guessed_item is not None,
            "decision_policy": "entropy_choice",
            "outcome_entropies": dict(outcome2entropy),
            "chosen_outcome": chosen_outcome,
            "tied_outcomes": tied_outcomes,
            "tie_break_order": list(tie_break_order),
            "result": chosen_outcome,
            "excluded_candidate": outcome2excluded.get(chosen_outcome),
            "entropy_before": entropy_before,
            "entropy_after": entropy_from_probs(chosen_post),
            "support_size_before": len(support_before),
            "support_size_after": len(support_after),
            "support_before": support_before[:20],
            "support_after": support_after[:20],
            "posterior_top1_candidate_before": top1_before,
            "posterior_top1_prob_before": top1p_before,
            "posterior_top1_candidate_after": top1_after,
            "posterior_top1_prob_after": top1p_after,
            "top_candidates_before": [
                {"candidate": c, "prob": p}
                for c, p in sorted(self.prior.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_candidates_after": [
                {"candidate": c, "prob": p}
                for c, p in sorted(chosen_post.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
        }

        self.prior = chosen_post
        self.turn_records.append(meta)
        self.last_response_type = "direct_guess"
        self.last_response_label = chosen_outcome
        return response_text

    def respond(self, text: str) -> str:
        if _is_question(text):
            return self.respond_to_question(text)
        return self.respond_to_guess(text)

    def export_state(self) -> Dict[str, Any]:
        support_final = self._support_candidates()
        top1_final, top1p_final = self._posterior_top1(self.prior)

        return {
            "mode": self.cfg.mode,
            "pool_name": self.cfg.pool_name,
            "final_target": self.final_target,
            "final_target_policy": self.final_target_policy,
            "final_entropy": entropy_from_probs(self.prior),
            "support_size_final": len(support_final),
            "support_final": support_final[:50],
            "posterior_top1_candidate_final": top1_final,
            "posterior_top1_prob_final": top1p_final,
            "termination_reason": self.termination_reason,
            "accepted_guess": self.accepted_guess,
            "last_response_type": self.last_response_type,
            "last_response_label": self.last_response_label,
            "effective_target_locked": len(support_final) == 1,
            "top_candidates_final": [
                {"candidate": c, "prob": p}
                for c, p in sorted(self.prior.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "turn_records": self.turn_records,
        }