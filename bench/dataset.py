"""A small, self-contained mixed benchmark.

Deliberately spans:
  - trivial/easy items where a cheap model ties Opus (proves the savings),
  - CRT "trap" items where single cheap models reliably fail but deliberation
    / Mixture-of-Agents recovers the right answer (proves the quality lever),
  - open-ended reasoning & code graded by an independent strong judge.

`type="exact"` items are graded by tolerant string/number matching.
`type="judge"` items are graded by an independent model against `reference`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    domain: str
    prompt: str
    type: str  # "exact" | "judge"
    answers: list[str] = field(default_factory=list)  # acceptable answers (exact)
    reference: str = ""  # rubric / reference (judge)


DATASET: list[Item] = [
    # ---- easy: cheap model should tie Opus ---------------------------------
    Item(
        "cap-au", "factual", "What is the capital of Australia? Answer with just the city name.", "exact", ["canberra"]
    ),
    Item("mult", "math", "What is 17 multiplied by 23? Give only the number.", "exact", ["391"]),
    Item("hexagon", "factual", "How many sides does a hexagon have? Number only.", "exact", ["6", "six"]),
    Item("mix-color", "factual", "What color results from mixing blue and yellow paint? One word.", "exact", ["green"]),
    Item(
        "speed",
        "math",
        "A train travels 60 km in 1.5 hours. What is its average speed in km/h? Number only.",
        "exact",
        ["40"],
    ),
    Item(
        "change",
        "math",
        "Pens cost $3 each. You buy 4 and pay with a $20 bill. How much change do you get in dollars? Number only.",
        "exact",
        ["8"],
    ),
    # ---- CRT "traps": single cheap models reliably get these wrong ----------
    Item(
        "bat-ball",
        "reasoning",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost, in cents? Give only the number of cents.",
        "exact",
        ["5", "5 cents", "$0.05", "0.05"],
    ),
    Item(
        "widgets",
        "reasoning",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines "
        "to make 100 widgets? Answer in minutes, number only.",
        "exact",
        ["5"],
    ),
    Item(
        "lilypad",
        "reasoning",
        "In a lake, a patch of lily pads doubles in size every day. If it takes 48 days for the "
        "patch to cover the entire lake, how many days would it take to cover half the lake? Number only.",
        "exact",
        ["47"],
    ),
    Item(
        "socks",
        "reasoning",
        "A drawer has 10 black and 10 white socks, drawn in the dark. What is the minimum number "
        "of socks you must take out to GUARANTEE a matching pair? Number only.",
        "exact",
        ["3"],
    ),
    # ---- medium/hard reasoning & math (exact) ------------------------------
    Item(
        "ages",
        "math",
        "Tom is twice as old as Sara was when Tom was as old as Sara is now. If Tom is 24, how old "
        "is Sara now? Number only.",
        "exact",
        ["18"],
    ),
    Item(
        "prob",
        "math",
        "Two fair six-sided dice are rolled. What is the probability the sum is 7? Answer as a simplified fraction.",
        "exact",
        ["1/6"],
    ),
    # ---- open-ended, judged ------------------------------------------------
    Item(
        "sqrt2",
        "reasoning",
        "Prove that the square root of 2 is irrational.",
        "judge",
        reference="A correct proof by contradiction: assume sqrt(2)=a/b in lowest terms, "
        "derive that a and b are both even, contradicting lowest terms.",
    ),
    Item(
        "palindrome",
        "code",
        "Write a Python function is_palindrome(s) that returns True if s is a palindrome, "
        "ignoring case, spaces, and punctuation. Include the function only.",
        "judge",
        reference="A correct function that normalizes (lowercases, strips non-alphanumerics) "
        "and compares the string to its reverse. Must handle 'A man, a plan, a canal: Panama' -> True.",
    ),
    Item(
        "bug",
        "code",
        "This Python is meant to return the average of a list but crashes on empty input:\n"
        "def avg(xs): return sum(xs)/len(xs)\n"
        "Fix it so an empty list returns 0.0, and explain the bug in one sentence.",
        "judge",
        reference="Must guard against len(xs)==0 (e.g., return 0.0 if not xs) and correctly "
        "identify division by zero on empty input.",
    ),
]
