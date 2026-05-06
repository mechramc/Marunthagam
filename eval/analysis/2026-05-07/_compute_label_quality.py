"""Compute label-quality stats from user-filled labels."""
import csv, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

labels = json.load(open("C:/Github/Marunthagam/eval/analysis/2026-05-07/_user_labels.json", encoding="utf-8"))
assert len(labels) == 80

with open("C:/Github/Marunthagam/eval/analysis/2026-05-07/label_quality_spotcheck.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 80

for row, (my, note) in zip(rows, labels):
    row["my_label"] = my
    row["notes"] = note

# Save labeled CSV
with open("C:/Github/Marunthagam/eval/analysis/2026-05-07/label_quality_spotcheck_LABELED.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Wrote labeled CSV.")
print()

# Stats
print("=== AGREEMENT BY (specialist, gold_label) ===")
buckets = {}
for r in rows:
    buckets.setdefault((r["specialist"], r["gold_label"]), []).append(r)

for (spec, gold), rs in sorted(buckets.items()):
    agree = sum(1 for r in rs if r["my_label"] == gold)
    n = len(rs)
    print(f"  {spec:8s} gold={gold:7s}  n={n:2d}  agree={agree:2d}  disagree={n-agree:2d}  ({agree/n*100:.0f}% agree)")

print()
print("=== DISAGREEMENT DIRECTION ===")
ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}
higher = lower = same = 0
for r in rows:
    if r["my_label"] == r["gold_label"]:
        continue
    my, gd = ORDER[r["my_label"]], ORDER[r["gold_label"]]
    if my > gd: higher += 1
    elif my < gd: lower += 1
    else: same += 1
total = higher + lower + same
print(f"  Total disagreements: {total}/80")
print(f"  Toward HIGHER acuity (under-triaged labels): {higher}")
print(f"  Toward LOWER acuity (over-triaged labels):   {lower}")
print(f"  Same level, different class:                {same}")

print()
print("=== WRONG-SPECIALIST ROUTING (notes mention) ===")
ws = [r for r in rows if "wrong specialist" in r["notes"].lower()]
print(f"  Total flagged: {len(ws)}/80")
ws_by_spec = {}
for r in ws:
    ws_by_spec.setdefault(r["specialist"], []).append(r)
for s, l in ws_by_spec.items():
    print(f"  {s}: {len(l)}/40 = {len(l)/40*100:.0f}%")

print()
print("=== PROJECTED RELABELED DISTRIBUTIONS (assuming sample rate generalises) ===")
# Triage: 90G / 209Y / 52R; GREEN agree=14/20=70%; YELLOW agree=20/20=100%; RED unsampled
greens_keep_t = round(90 * 14/20)
greens_to_y_t = 90 - greens_keep_t
print(f"  Triage train (current 90G/209Y/52R, n=351):")
print(f"    GREEN agreement {14}/20 = 70%. Projected GREEN→YELLOW shift: ~{greens_to_y_t} of 90.")
print(f"    Projected: {greens_keep_t}G / {209+greens_to_y_t}Y / 52R")
print(f"    Old %: 25.6 / 59.5 / 14.8 → New %: {greens_keep_t/351*100:.1f} / {(209+greens_to_y_t)/351*100:.1f} / {52/351*100:.1f}")

# Derm
greens_keep_d = round(142 * 16/20)
greens_to_y_d = 142 - greens_keep_d
yellows_keep_d = round(164 * 18/20)
yellows_to_r_d = 164 - yellows_keep_d
print(f"  Derm train (current 142G/164Y/22R, n=328):")
print(f"    GREEN agreement {16}/20 = 80%. Projected GREEN→YELLOW shift: ~{greens_to_y_d} of 142.")
print(f"    YELLOW agreement {18}/20 = 90%. Projected YELLOW→RED shift: ~{yellows_to_r_d} of 164.")
print(f"    Projected: {greens_keep_d}G / {yellows_keep_d + greens_to_y_d}Y / {22 + yellows_to_r_d}R")

print()
print("=== CRITICAL OBSERVATIONS ===")
print("1. 100% of disagreements are in the UNDER-TRIAGED direction. The dataset systematically")
print("   labels things at lower acuity than the rater (clinician).")
print("2. Triage YELLOW labels are CLEAN (20/20 agreement). The noise is concentrated in GREEN.")
print("3. Derm has wrong-specialist routing contamination — at least", len(ws_by_spec.get("derm", [])), "of 40 derm cases (",
      f"{len(ws_by_spec.get('derm', []))/40*100:.0f}%) are not derm cases at all (poison, hepatology, pulmonology).")
print("4. After projected relabeling, triage YELLOW prior would INCREASE from 60% to ~67%.")
print("   Class-balanced retraining on noisy labels would push the model AWAY from clinical")
print("   correctness. Relabeling first is the right sequence.")
