from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from reasoning.evaluate.twenty_question.offline_evaluator import (
        POOL_MAP,
        LocalVLLMJudge,
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


def _extract_guess(text: str) -> str:
    """
    Extract guessed entity only for explicit direct-guess style utterances.
    We do NOT treat ordinary questions like 'Is X a dog?' as a direct guess.
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()

    patterns = [
        r'^\s*x\s+is\s+["\']?(.+?)["\']?\s*$',
        r'^\s*my\s+guess\s+is\s+["\']?(.+?)["\']?\s*$',
        r'^\s*the\s+answer\s+is\s+["\']?(.+?)["\']?\s*$',
        r'^\s*i\s+think\s+it\s+is\s+["\']?(.+?)["\']?\s*$',
        r'^\s*it\s+is\s+["\']?(.+?)["\']?\s*$',
    ]
    for p in patterns:
        m = re.match(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def _normalize_surface(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


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

    def __init__(self, cfg: OracleConfig, judge: LocalVLLMJudge):
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

    def _answer_map(self, question: str, active_candidates: Sequence[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        bs = max(1, self.cfg.judge_batch_size)

        for i in range(0, len(active_candidates), bs):
            batch = list(active_candidates[i:i + bs])
            batch_out = self.judge.judge_batch(batch, question)
            result.update(batch_out)

        return result

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

        answer2posterior: Dict[str, Dict[str, float]] = {}
        answer2entropy: Dict[str, float] = {}

        for ans in ("yes", "no", "pass"):
            post = self._posterior_given_answer(self.prior, answer_map, ans)
            h = entropy_from_probs(post)
            answer2posterior[ans] = post
            answer2entropy[ans] = h

        if self.cfg.mode == "adv":
            chosen_answer = max(("yes", "no", "pass"), key=lambda a: answer2entropy[a])
        else:
            chosen_answer = min(("yes", "no", "pass"), key=lambda a: answer2entropy[a])

        chosen_post = answer2posterior[chosen_answer]

        support_after = self._support_candidates_from_prior(chosen_post)
        top1_after, top1p_after = self._posterior_top1(chosen_post)

        meta = {
            "type": "question",
            "question": question,
            "entropy_before": entropy_from_probs(self.prior),
            "answer_entropies": dict(answer2entropy),
            "chosen_answer": chosen_answer,
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

        if chosen_answer == "yes":
            return "Yes."
        if chosen_answer == "no":
            return "No."
        return "Pass."

    def respond_to_guess(self, text: str) -> str:
        guess_raw = _extract_guess(text)
        guess_norm = normalize_entity(guess_raw)

        support_before = self._support_candidates()
        top1_before, top1p_before = self._posterior_top1(self.prior)
        entropy_before = entropy_from_probs(self.prior)

        prior_new = dict(self.prior)
        excluded_candidate = None
        guessed_item = None

        support_norm_to_item = {normalize_entity(c): c for c in support_before}
        guessed_item = support_norm_to_item.get(guess_norm, None)

        if self.cfg.mode == "cop":
            if guessed_item is not None:
                self.final_target = guessed_item
                meta = {
                    "type": "direct_guess",
                    "guess_raw": guess_raw,
                    "guessed_entity": guessed_item,
                    "result": "correct",
                    "excluded_candidate": None,
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_before,
                    "support_size_before": len(support_before),
                    "support_size_after": len(support_before),
                    "posterior_top1_candidate_before": top1_before,
                    "posterior_top1_prob_before": top1p_before,
                    "posterior_top1_candidate_after": top1_before,
                    "posterior_top1_prob_after": top1p_before,
                }
                self.turn_records.append(meta)
                return "Correct, you guessed it."

            if guessed_item is None:
                for c in list(prior_new.keys()):
                    if normalize_entity(c) == guess_norm:
                        excluded_candidate = c
                        prior_new[c] = 0.0
                        break
                prior_new = normalize_probs(prior_new)

            self.prior = prior_new
            support_after = self._support_candidates()
            top1_after, top1p_after = self._posterior_top1(self.prior)

            if support_after:
                self.final_target = max(support_after, key=lambda c: self.prior.get(c, 0.0))
            else:
                self.final_target = None

            meta = {
                "type": "direct_guess",
                "guess_raw": guess_raw,
                "guessed_entity": guessed_item if guessed_item is not None else guess_raw,
                "result": "incorrect",
                "excluded_candidate": excluded_candidate,
                "entropy_before": entropy_before,
                "entropy_after": entropy_from_probs(self.prior),
                "support_size_before": len(support_before),
                "support_size_after": len(support_after),
                "posterior_top1_candidate_before": top1_before,
                "posterior_top1_prob_before": top1p_before,
                "posterior_top1_candidate_after": top1_after,
                "posterior_top1_prob_after": top1p_after,
            }
            self.turn_records.append(meta)
            return "No, that is not correct."

        # adv mode
        if guessed_item is not None and len(support_before) == 1:
            self.final_target = guessed_item
            meta = {
                "type": "direct_guess",
                "guess_raw": guess_raw,
                "guessed_entity": guessed_item,
                "result": "correct",
                "excluded_candidate": None,
                "entropy_before": entropy_before,
                "entropy_after": entropy_before,
                "support_size_before": len(support_before),
                "support_size_after": len(support_before),
                "posterior_top1_candidate_before": top1_before,
                "posterior_top1_prob_before": top1p_before,
                "posterior_top1_candidate_after": top1_before,
                "posterior_top1_prob_after": top1p_before,
            }
            self.turn_records.append(meta)
            return "Correct, you guessed it."

        # incorrect guess should only exclude that guessed item if it is in current support
        if guessed_item is not None:
            excluded_candidate = guessed_item
            prior_new[guessed_item] = 0.0
        else:
            # if the guessed entity is not in current support, do not mutate posterior
            excluded_candidate = None

        prior_new = normalize_probs(prior_new)
        self.prior = prior_new

        support_after = self._support_candidates()
        top1_after, top1p_after = self._posterior_top1(self.prior)

        if support_after:
            for c in support_after:
                if normalize_entity(c) != guess_norm:
                    self.final_target = c
                    break
            if self.final_target is None:
                self.final_target = support_after[0]
        else:
            self.final_target = None

        meta = {
            "type": "direct_guess",
            "guess_raw": guess_raw,
            "guessed_entity": guessed_item if guessed_item is not None else guess_raw,
            "result": "incorrect",
            "excluded_candidate": excluded_candidate,
            "entropy_before": entropy_before,
            "entropy_after": entropy_from_probs(self.prior),
            "support_size_before": len(support_before),
            "support_size_after": len(support_after),
            "posterior_top1_candidate_before": top1_before,
            "posterior_top1_prob_before": top1p_before,
            "posterior_top1_candidate_after": top1_after,
            "posterior_top1_prob_after": top1p_after,
        }
        self.turn_records.append(meta)
        return "No, that is not correct."

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
            "final_entropy": entropy_from_probs(self.prior),
            "support_size_final": len(support_final),
            "support_final": support_final[:50],
            "posterior_top1_candidate_final": top1_final,
            "posterior_top1_prob_final": top1p_final,
            "top_candidates_final": [
                {"candidate": c, "prob": p}
                for c, p in sorted(self.prior.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "turn_records": self.turn_records,
        }