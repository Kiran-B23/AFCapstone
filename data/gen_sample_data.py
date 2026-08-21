"""Generate the sample dataset used by AdvisorIQ.

Run once:  python data/gen_sample_data.py
Produces:
  data/sample_transactions.csv   -> labeled transactions for the fraud/anomaly checker
  data/docs/*.txt                -> tiny finance "research library" for RAG

Deterministic (fixed seed) so results are reproducible and the smoke test is stable.
"""
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")

MERCHANTS = {
    "Amazon": 0.1, "Walmart": 0.1, "Starbucks": 0.1, "Uber": 0.2, "Apple": 0.1,
    "SteamGames": 0.3, "CryptoX": 0.9, "GiftCardHub": 0.85, "WireTransferCo": 0.8,
    "Netflix": 0.1, "Shell": 0.15, "UnknownIntl": 0.95,
}
COUNTRIES = ["US", "US", "US", "US", "UK", "IN", "NG", "RU"]  # weighted toward US


def _make_row(i, rng):
    merchant = rng.choice(list(MERCHANTS))
    merchant_risk = MERCHANTS[merchant]
    country = rng.choice(COUNTRIES)
    is_foreign = 0 if country == "US" else 1
    hour = rng.randint(0, 23)
    # amount: mostly small, occasionally large
    amount = round(abs(rng.gauss(80, 60)) + (rng.random() < 0.1) * rng.uniform(500, 4000), 2)

    # "true" suspicion signal (what the model must learn): large + foreign + risky merchant + odd hour
    score = (
        0.4 * (amount > 800)
        + 0.25 * is_foreign
        + 0.30 * (merchant_risk > 0.7)
        + 0.15 * (hour < 5 or hour > 22)
    )
    is_suspicious = int(score > 0.5 or (score > 0.35 and rng.random() < 0.3))

    return {
        "txn_id": f"T{i:05d}",
        "hour": hour,
        "amount": amount,
        "merchant": merchant,
        "merchant_risk": merchant_risk,
        "country": country,
        "is_foreign": is_foreign,
        "account_number": f"{rng.randint(4000_0000_0000_0000, 4999_9999_9999_9999)}",  # PII on purpose
        "is_suspicious": is_suspicious,
    }


def main(n=600, seed=42):
    rng = random.Random(seed)
    rows = [_make_row(i, rng) for i in range(n)]
    out = os.path.join(HERE, "sample_transactions.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pos = sum(r["is_suspicious"] for r in rows)
    print(f"Wrote {out}: {n} rows, {pos} suspicious ({pos/n:.0%})")

    os.makedirs(DOCS_DIR, exist_ok=True)
    for name, text in DOCS.items():
        with open(os.path.join(DOCS_DIR, name), "w") as f:
            f.write(text.strip() + "\n")
    print(f"Wrote {len(DOCS)} docs to {DOCS_DIR}")


# --- tiny research library (educational, generic; not investment advice) ---
DOCS = {
    "apple.txt": """
Apple Inc. (AAPL) overview.
Apple designs consumer electronics (iPhone, Mac, iPad), wearables, and a growing services
business (App Store, iCloud, Apple Music). Services revenue has become a larger share of
total revenue and carries higher margins than hardware. Key risks include dependence on
iPhone sales, supply-chain concentration, and regulatory scrutiny of the App Store.
Investors often watch iPhone unit trends, services growth rate, and gross margin.
""",
    "microsoft.txt": """
Microsoft Corp (MSFT) overview.
Microsoft's revenue comes from productivity software (Office/Microsoft 365), cloud
infrastructure (Azure), Windows, and gaming. Azure growth and enterprise cloud adoption
are the most-watched drivers. Risks include cloud competition and large capital spending
on data centers. Recurring subscription revenue makes results relatively stable.
""",
    "diversification.txt": """
Diversification basics.
Diversification means spreading investments across different assets, sectors, and regions
so that a loss in one holding does not sink the whole portfolio. A concentrated portfolio
(one or two stocks) has higher risk. Index funds are a common low-cost way to diversify.
Past performance does not guarantee future results, and all investing involves risk.
""",
    "risk.txt": """
Risk and volatility.
Volatility measures how much a price moves up and down over time. Higher volatility means
larger swings and higher risk. Diversification, a long time horizon, and position sizing
are common ways to manage risk. Never invest money you cannot afford to lose. There is no
such thing as a guaranteed return in the stock market.
""",
    "fraud_signs.txt": """
Signs of suspicious transactions.
Common red flags include unusually large amounts, transactions in foreign countries you did
not visit, purchases at odd hours, high-risk merchants (gift cards, crypto, wire transfers),
and rapid repeated charges. If several red flags appear together, the transaction is more
likely to be fraudulent and should be reviewed.
""",
}


if __name__ == "__main__":
    main()
