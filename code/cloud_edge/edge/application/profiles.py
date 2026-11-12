"""Independent arms; configured BCCL is the single shared baseline."""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
from .config import algorithm_hash


@dataclass(frozen=True)
class Profile:
    arm: str
    method: str
    p_low: float
    tau_a: float
    alpha: float
    bank: bool = True
    replay: int = 80
    family: str = "methods"

    def to_dict(self):
        return asdict(self)


def profile_book(raw):
    c = raw["continuous"]
    p, t, a = (float(c[k]) for k in ("p_low", "non_lie_ratio_threshold", "alpha"))
    replay = int(raw["data_mix"]["source_sample_count"])
    book = {m: Profile(m, m, p, t, a, True, replay) for m in
            ("full_upload", "confidence", "activity", "bccl")}
    book["A_source_only"] = Profile("A_source_only", "source_only", p, t, a, False, 0, "ablation")
    for name, bank in (("D_nlr03_nobank", False), ("D_nlr03", True)):
        book[name] = Profile(name, "bccl", p, t, a, bank, 0, "ablation")
    for pq in (10, 20, 30, 40, 50):
        for ta in (10, 20, 30, 40, 50):
            name = f"D_grid_pq{pq:02d}_ta{ta:02d}"
            book[name] = Profile(name, "bccl", pq / 100, ta / 100, .05, True, replay, "joint")
    for alpha in (1, 5, 10, 20):
        name = f"D_alpha{alpha:02d}"
        book[name] = Profile(name, "bccl", p, t, alpha / 100, True, replay, "alpha")
    return book


def canonical(name, raw):
    aliases = {"BCCL": "bccl", "source_only": "A_source_only", "D_nlr03_replay": "bccl",
               "entropy": "activity", "non_lie": "activity"}
    name = aliases.get(name, name)
    book = profile_book(raw)
    if name not in book:
        raise ValueError(f"Unknown arm: {name}")
    profile = book[name]
    base = book["bccl"]
    if (profile.family in ("joint", "alpha") and
            (profile.p_low, profile.tau_a, profile.alpha) == (base.p_low, base.tau_a, base.alpha)):
        return base
    return profile


def select_profiles(raw, suite="methods", names=None):
    if names:
        selected = names
    else:
        book = profile_book(raw)
        suites = {
            "methods": ["full_upload", "confidence", "activity", "bccl"],
            "ablation": ["A_source_only", "D_nlr03_nobank", "D_nlr03", "bccl"],
            "joint": ["bccl"] + [p.arm for p in book.values() if p.family == "joint"],
            "alpha": ["bccl"] + [p.arm for p in book.values() if p.family == "alpha"],
        }
        if suite == "all":
            selected = [name for part in suites.values() for name in part]
        elif suite in suites:
            selected = suites[suite]
        else:
            raise ValueError(f"Unknown suite: {suite}")
    result = {}
    for name in selected:
        p = canonical(name, raw)
        result[p.arm] = p
    return list(result.values())


def apply_profile(shared, profile):
    raw = copy.deepcopy(shared)
    raw["continuous"].update(method=profile.method, p_low=profile.p_low,
                              non_lie_ratio_threshold=profile.tau_a, alpha=profile.alpha)
    raw["data_mix"].update(source_sample_count=profile.replay,
                            mix_mode="source_plus_hard" if profile.replay else "hard_only")
    raw["model_bank"].update(enabled=profile.bank, query_on_train_start=profile.bank,
                              rejection_fallback=profile.bank, deploy_competition=False)
    raw["continuous_ablation_group"] = profile.arm
    raw["continuous_sensitivity_family"] = profile.family
    return raw


def profile_hash(shared, profile):
    return algorithm_hash(apply_profile(shared, profile))
