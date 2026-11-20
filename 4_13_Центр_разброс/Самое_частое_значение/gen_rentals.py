#!/usr/bin/env python3
"""Generate rentals.csv with 30 rows and controlled modes.

Target modes (must match what the Rust program prints):
  item      -> ski         (18 of 30)
  boot_size -> 42          (7  of 30)
  level     -> intermediate(12 of 30)
  days      -> 3           (10 of 30)
  mean(days)= 95/30 = 3.1666... -> round(1) = 3.2   (distinct from mode 3)
"""
import csv
from collections import Counter

# --- per-column value lists with exact frequencies ----------------------
items = ["ski"] * 18 + ["snowboard"] * 12                 # 30, mode ski (18)
boot_sizes = (
    [38] * 2 + [39] * 3 + [40] * 4 + [41] * 5 +
    [42] * 7 + [43] * 4 + [44] * 3 + [45] * 2
)                                                         # 30, mode 42 (7)
levels = (
    ["beginner"] * 8 + ["intermediate"] * 12 + ["advanced"] * 10
)                                                         # 30, mode intermediate (12)
days = (
    [1] * 4 + [2] * 6 + [3] * 10 + [4] * 5 + [5] * 3 + [7] * 2
)                                                         # 30, mode 3 (10); sum 95

assert len(items) == 30
assert len(boot_sizes) == 30
assert len(levels) == 30
assert len(days) == 30

# Pair each rental so a larger boot / advanced level tends to rent longer:
# interleave deterministically but keep the exact per-column frequencies.
# We sort days descending and boot descending so they correlate (more realistic
# for the ML model), then attach items/levels round-robin-ish.
order = sorted(range(30), key=lambda i: (-boot_sizes[i], -days[i]))
rows = []
for k, i in enumerate(order):
    rows.append((items[i], boot_sizes[i], levels[i], days[i]))

with open("/home/z/my-project/rust_rentals/rentals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "boot_size", "level", "days"])
    for r in rows:
        w.writerow(r)

# --- self-check ---------------------------------------------------------
def mode_count(seq):
    c = Counter(seq)
    top = max(c.values())
    # pandas.mode()[0] -> smallest value among the most frequent
    return min(v for v, n in c.items() if n == top), top

for name, seq in [("item", items), ("boot_size", boot_sizes),
                  ("level", levels), ("days", days)]:
    m, n = mode_count(seq)
    print(f"{name}: {m} ({n} of 30)")

mean = sum(days) / len(days)
print(f"Days rented, mean vs mode: {round(mean, 1)} vs {mode_count(days)[0]}")
print("CSV written:", "/home/z/my-project/rust_rentals/rentals.csv")
