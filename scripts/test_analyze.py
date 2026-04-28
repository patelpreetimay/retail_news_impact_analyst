"""Quick test of the /analyze endpoint."""
import requests, json

URL = "http://127.0.0.1:8000/analyze"

tests = {
    "Earnings": "NVIDIA reported Q4 revenue of 22.1 billion, up 265 percent year-over-year, smashing analyst estimates of 20.4 billion. Data center revenue alone hit 18.4 billion.",
    "M&A": "Reuters has learned that Apple is in advanced talks to acquire London-based AI startup for approximately 4.2 billion. The deal would mark Apple largest acquisition since Beats.",
    "Regulatory": "The US Securities and Exchange Commission today opened a formal investigation into Tesla accounting practices around its Full Self-Driving revenue recognition.",
    "Bug headline": "Stocks to buy or sell: Osho Krishan of Angel One suggests buying Chennai Petroleum Corp, Bharti Airtel shares to buy",
    "Gibberish": "vch jhabcuyabc abchaubc ajgua ahbu",
    "Gibberish2": "sdbvkisd bcija abuja anci anc",
}

for name, text in tests.items():
    r = requests.post(URL, json={"text": text})
    d = r.json()
    if r.status_code == 200:
        print(f"[OK]  {name}: event={d['event_type']}, stance={d['stance']}, impact={d['impact_score']:.2f}, conf={d.get('event_confidence','?')}")
    else:
        print(f"[REJ] {name}: {r.status_code} — {d.get('detail','?')}")
