"""
Acquire English medical Q&A source corpora and bucket into triage/derm/maternal.

Source: lavita/ChatDoctor-HealthCareMagic-100k (consumer patient Q&A).
Buckets are assigned by keyword priority: derm > maternal > triage.

Usage:
    python acquire_sources.py --per-specialist 500 --seed 42
    python acquire_sources.py --per-specialist 2000 --triage-cap 2000 \
        --derm-cap 1200 --maternal-cap 1500
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

TRAINING_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = TRAINING_ROOT / "data" / "raw"

DERM_KEYWORDS = re.compile(
    r"\b(skin|rash|rashes|eczema|acne|pimple|pimples|mole|moles|derm|scalp|"
    r"itch|itchy|itching|lesion|lesions|dermatitis|psoriasis|hives|"
    r"ringworm|fungal|wart|warts|blister|blisters|boil|boils|"
    r"birthmark|melanoma|fitzpatrick)\b",
    re.IGNORECASE,
)

MATERNAL_KEYWORDS = re.compile(
    r"\b(pregnan|pregnancy|prenatal|antenatal|postpartum|fetus|fetal|"
    r"miscarriage|breastfeed|breastfeeding|lactation|contraception|"
    r"contraceptive|menstrual|menstruation|period|periods|ovulat|"
    r"newborn|infant|baby|babies|umbilical|nipple|labor|labour|"
    r"obstetric|gestation|trimester|gynec|gynaec|cervix|uterus|uterine|"
    r"placenta|amniotic)\b",
    re.IGNORECASE,
)


def bucket_for(text: str) -> str:
    """Priority-ordered bucket: derm > maternal > triage."""
    if DERM_KEYWORDS.search(text):
        return "derm"
    if MATERNAL_KEYWORDS.search(text):
        return "maternal"
    return "triage"


def is_acceptable_pair(question: str, answer: str) -> bool:
    """Drop empty, very short, or obviously non-medical rows."""
    if not question or not answer:
        return False
    if len(question.strip()) < 30 or len(answer.strip()) < 30:
        return False
    return True


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def acquire(
    triage_cap: int,
    derm_cap: int,
    maternal_cap: int,
    seed: int,
) -> dict[str, int]:
    """Stream ChatDoctor, bucket rows, write per-specialist JSONL up to caps."""
    rng = random.Random(seed)
    caps = {"triage": triage_cap, "derm": derm_cap, "maternal": maternal_cap}
    buckets: dict[str, list[dict]] = {"triage": [], "derm": [], "maternal": []}

    print("Streaming lavita/ChatDoctor-HealthCareMagic-100k ...")
    ds = load_dataset(
        "lavita/ChatDoctor-HealthCareMagic-100k",
        split="train",
        streaming=True,
    )

    seen = 0
    for example in ds:
        seen += 1
        question = (example.get("input") or "").strip()
        answer = (example.get("output") or "").strip()
        if not is_acceptable_pair(question, answer):
            continue

        bucket = bucket_for(question + " " + answer)
        if len(buckets[bucket]) >= caps[bucket]:
            if all(len(buckets[k]) >= caps[k] for k in caps):
                break
            continue

        buckets[bucket].append(
            {
                "question": question,
                "answer": answer,
                "source": "chatdoctor-healthcaremagic-100k",
            }
        )

        if seen % 5000 == 0:
            sizes = ", ".join(f"{k}={len(v)}/{caps[k]}" for k, v in buckets.items())
            print(f"  scanned={seen}  {sizes}")

    sizes_out = {}
    for specialist, rows in buckets.items():
        rng.shuffle(rows)
        out_path = RAW_DIR / f"{specialist}_en.jsonl"
        write_jsonl(out_path, rows)
        sizes_out[specialist] = len(rows)
        print(f"Wrote {len(rows):>5} rows -> {out_path}")

    return sizes_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire English medical Q&A corpora")
    parser.add_argument(
        "--per-specialist",
        type=int,
        default=None,
        help="Shorthand cap applied to all three specialists",
    )
    parser.add_argument("--triage-cap", type=int, default=2000)
    parser.add_argument("--derm-cap", type=int, default=1200)
    parser.add_argument("--maternal-cap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_specialist is not None:
        triage_cap = derm_cap = maternal_cap = args.per_specialist
    else:
        triage_cap = args.triage_cap
        derm_cap = args.derm_cap
        maternal_cap = args.maternal_cap

    sizes = acquire(triage_cap, derm_cap, maternal_cap, args.seed)
    total = sum(sizes.values())
    print(f"\nDone. Total: {total} rows across {len(sizes)} specialists")


if __name__ == "__main__":
    main()
