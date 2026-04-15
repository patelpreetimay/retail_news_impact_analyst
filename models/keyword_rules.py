"""
keyword_rules.py — Domain-Specific Keyword Rule Classifiers for RNIA
=====================================================================

Simple pattern-matching classifiers that use curated keyword dictionaries
to predict event type and stance.  Designed to be ensembled with the
TF-IDF+LR models for complementary signal.

These classifiers excel at high-precision calls on "obvious" articles
(e.g. "CEO resigns" → Leadership_Change) and provide a useful
regularisation signal when combined with the statistical model.
"""

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event keyword dictionaries
# ---------------------------------------------------------------------------

EVENT_KEYWORDS: dict[str, list[str]] = {
    "Earnings": [
        r"\bearnings?\b", r"\brevenue\b", r"\bprofit(?:s|ability)?\b",
        r"\bquarterly results?\b", r"\beps\b", r"\bdividend\b",
        r"\bfinancial results?\b", r"\bnet income\b", r"\bgross margin\b",
        r"\boperating income\b", r"\bfiscal\b",
        r"\bbeat expectations?\b", r"\bmissed expectations?\b", r"\bguidance\b",
        r"\bsales growth\b", r"\bsame.store sales\b", r"\bcomps\b",
        r"\btop.?line\b", r"\bbottom.?line\b", r"\bebitda\b",
        r"\bfree cash flow\b", r"\bfcf\b", r"\bmargins?\b",
        r"\bprofit warning\b", r"\brestate(?:d|ment)?\b",
        r"\bforecast\b", r"\boutlook\b", r"\byear[- ]over[- ]year\b",
        r"\byoy\b", r"\bqoq\b", r"\bquarter[- ]on[- ]quarter\b",
        r"\bq[1-4]\b", r"\bh[12]\b", r"\bfy\d{2,4}\b",
        r"\bbookings?\b", r"\bbillings?\b", r"\bbacklog\b",
    ],
    "Leadership_Change": [
        r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b", r"\bcio\b",
        r"\bchief (?:executive|financial|operating|technology|information) officer\b",
        r"\bappointed?\b", r"\bresign(?:ed|ation|s)?\b",
        r"\bboard of directors\b", r"\bexecutive.{0,40}?(?:change|shakeup|transition)\b",
        r"\bstepping down\b", r"\bsuccession\b",
        r"\bnew.{0,40}?(?:ceo|chief|president|chairman)\b",
        r"\bouster\b", r"\bousted\b", r"\bfired\b", r"\bterminated\b",
        r"\bleadership (?:change|transition|shakeup)\b",
        r"\breshuffl(?:e|ed|ing)\b", r"\bdeparture\b", r"\bexits\b",
        r"\bjoins as\b", r"\binterim ceo\b", r"\bnamed (?:ceo|cfo|chairman)\b",
    ],
    "Regulatory_Action": [
        r"\bregulat(?:ion|ory|or|ed|ions)\b",
        r"\bsec\b", r"\bftc\b", r"\bfda\b", r"\bdoj\b", r"\bcftc\b",
        r"\brbi\b", r"\bsebi\b", r"\bcci\b", r"\birdai\b",
        r"\bsecurities and exchange commission\b",
        r"\bcompliance\b", r"\bfine[ds]?\b", r"\bpenalt(?:y|ies|ised|ized)\b",
        r"\banti.?trust\b", r"\bmonopoly\b", r"\bsanction(?:s|ed|ing)?\b",
        r"\bpolicy\b", r"\blegislat(?:ion|ive|ors?|ed)\b",
        r"\bgovernment.{0,40}?(?:action|order|crackdown)\b",
        r"\bban(?:ned|s|ning)?\b", r"\brestriction\b",
        r"\binvestigat(?:e[ds]?|ion|ing|ions)\b",  # moved here from Legal_Action
        r"\bprob(?:e[ds]?|ing)\b",
        r"\bsubpoena(?:s|ed)?\b", r"\baudit(?:s|ed|ing)?\b",
        r"\boversight\b", r"\bruling\b", r"\bmandate[ds]?\b",
        r"\bcease and desist\b", r"\bconsent decree\b",
        r"\bsuspended?\b", r"\bhalt(?:ed|s|ing)?\b",
        r"\bdpiit\b", r"\bgst\b", r"\btax (?:notice|raid|order)\b",
    ],
    "Mergers_Acquisitions": [
        r"\bmerger\b", r"\bacquisition\b", r"\btakeover\b", r"\bbuyout\b",
        r"\bdeal\b", r"\bjoint venture\b", r"\bpartnership\b",
        r"\bacquir(?:e[ds]?|ing)\b", r"\bmerg(?:e[ds]?|ing)\b",
        r"\bipo\b", r"\bspinoff\b", r"\bspin.off\b", r"\bdivestiture\b",
        r"\bbid (?:for|to acquire|of)\b", r"\bbidder\b",
        r"\bacquired by\b", r"\btarget of\b",
        r"\ball.cash deal\b", r"\ball.stock deal\b",
        r"\bgoing private\b", r"\bleveraged buyout\b", r"\blbo\b",
        r"\bprivate equity\b", r"\bhostile takeover\b",
        r"\bdue diligence\b", r"\bdefinitive agreement\b",
        r"\bstake (?:in|of)\b", r"\bcontrolling stake\b",
        r"\bequity infusion\b",
    ],
    "Legal_Action": [
        r"\blawsuit\b", r"\bsue[ds]?\b", r"\bcourt\b", r"\blitigation\b",
        r"\bsettlement\b", r"\bverdict\b", r"\binjunction\b",
        r"\bclass.action\b", r"\bpatent.{0,40}?(?:infringement|violation)\b",
        r"\bfraud\b",
        r"\bindict(?:ment|ed)\b", r"\barbitration\b",
        r"\bplaintiff\b", r"\bdefendant\b",
        r"\ballegations?\b", r"\bcharge[ds]?\b",
        r"\bprosecut(?:e[ds]?|ion|or)\b", r"\bcrimina(?:l|lly)\b",
        r"\bcivil suit\b", r"\bdeposition\b", r"\btrial\b",
        r"\bguilty\b", r"\bacquitt(?:al|ed)\b",
        r"\bbankrupt(?:cy)?\b", r"\bchapter\s+(?:7|11|13)\b",
        r"\binsolvency\b", r"\bliquidation\b", r"\breceivership\b",
    ],
    "Product_Announcement": [
        r"\blaunch(?:ed|es|ing)?\b", r"\breleas(?:e[ds]?|ing)\b",
        r"\bnew product\b", r"\binnovation\b", r"\broll.?out\b",
        r"\bprototype\b", r"\bnew model\b", r"\bfeature[ds]?\b",
        r"\bupgrade\b", r"\bnext.gen(?:eration)?\b",
        r"\bunveil(?:s|ed|ing)?\b", r"\bdebut(?:s|ed|ing)?\b",
        r"\bships?\b", r"\bbeta\b", r"\bpre.?order\b",
        r"\bavailable (?:from|now|today)\b",
        r"\bnew (?:line|series|version)\b",
        r"\brefresh(?:ed)?\b", r"\bredesign(?:ed)?\b",
        r"\b(?:hardware|software) launch\b",
    ],
    "Market_Movement": [
        # Macroeconomic / Geopolitical
        r"\bgdp\b", r"\binflation\b", r"\binterest rate\b", r"\bfed(?:eral reserve)?\b",
        r"\btariff\b", r"\btrade war\b", r"\bgeopolitic\b", r"\bsupply chain\b",
        r"\brecession\b", r"\bemployment\b", r"\bunemployment\b", r"\bjobs report\b",
        r"\bcentral bank\b", r"\bmonetary policy\b", r"\bfiscal policy\b",
        r"\bcurrency\b", r"\bexchange rate\b", r"\boil price\b",
        r"\bglobal.{0,40}?(?:economy|growth|trade)\b", r"\bwar\b", r"\bconflict\b",
        r"\brate (?:cut|hike|hold|pause|decision)\b",
        r"\bbasis points?\b", r"\bbps\b",
        r"\brupee\b", r"\bdollar\b", r"\beuro\b", r"\byen\b", r"\byuan\b",
        # Indices
        r"\bsensex\b", r"\bnifty\b", r"\bdow\b", r"\bnasdaq\b",
        r"\bs&p\b", r"\bftse\b", r"\bdax\b", r"\bnikkei\b", r"\bhang seng\b",
        # Market Sentiment / Investor Action
        r"\brally\b", r"\bsell.off\b", r"\binvestor\b", r"\banalyst\b",
        r"\bupgrade[ds]?\b", r"\bdowngrade[ds]?\b", r"\btarget price\b",
        r"\bshort.?sell\b", r"\bbuyback\b", r"\bshare repurchase\b",
        r"\bmarket.{0,40}?(?:sentiment|rally|crash|correction)\b",
        r"\bbull(?:ish)?\b", r"\bbear(?:ish)?\b", r"\bvolatil(?:e|ity)\b",
        r"\bhedge fund\b", r"\binstitutional\b",
        r"\bfii\b", r"\bdii\b", r"\bforeign investor\b", r"\bretail investor\b",
        r"\bcorrection\b", r"\bbull run\b", r"\bbear market\b",
        # Other / General
        r"\bcorporate social\b", r"\besg\b", r"\bsustainab\b",
        r"\bcharity\b", r"\bphilanthropy\b",
    ],
}

# Compile patterns for speed
_COMPILED_EVENT: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in EVENT_KEYWORDS.items()
}

# ---------------------------------------------------------------------------
# Stance keyword dictionaries
# ---------------------------------------------------------------------------

STANCE_KEYWORDS: dict[str, list[str]] = {
    "bullish": [
        r"\bgrowth\b", r"\bbeat\b", r"\brecord\b", r"\bsurge[ds]?\b",
        r"\bupgrade[ds]?\b", r"\bexpansion\b", r"\boptimis(?:m|tic)\b",
        r"\bstrong\b", r"\braise[ds]?\b", r"\bpositive\b", r"\bgain[ds]?\b",
        r"\brally\b", r"\bsoar(?:ed|ing|s)?\b", r"\bimproved?\b",
        r"\brecovery\b", r"\boutperform\b", r"\bexceed(?:s|ed|ing)?\b",
        r"\bpromising\b", r"\brobust\b", r"\bupside\b",
        r"\bconfiden(?:t|ce)\b",
        r"\bjump(?:ed|s|ing)?\b", r"\bclimb(?:ed|s|ing)?\b",
        # Note: "advanced" deliberately omitted — too often appears in
        # procedural M&A language ("advanced talks") rather than as sentiment.
        r"\baccelerat(?:e|ed|ing)\b",
        r"\bmilestone\b", r"\ball.time high\b", r"\brecord high\b",
        r"\bbreakthrough\b", r"\bmomentum\b", r"\bhealthy\b",
        r"\bsmash(?:ed|es|ing)?\b", r"\bblockbuster\b", r"\bblowout\b",
        r"\bstellar\b", r"\bbanner (?:quarter|year)\b",
        r"\bdouble[ -]digit growth\b", r"\bsoaring\b", r"\bsurging\b",
    ],
    "bearish": [
        r"\bdecline[ds]?\b", r"\bmiss(?:ed|es)?\b", r"\bplunge[ds]?\b",
        r"\bdowngrade[ds]?\b", r"\blayoff(?:s|ed)?\b", r"\bwarning\b",
        r"\bweak(?:ness|ened?)?\b", r"\bcut[ts]?\b", r"\bdrop(?:ped|s)?\b",
        r"\bloss(?:es)?\b", r"\bnegative\b", r"\bpessimis(?:m|tic)\b",
        r"\bfall(?:en|s|ing)?\b", r"\bslump(?:ed|ing|s)?\b", r"\bcrash(?:ed|es|ing)?\b",
        r"\bshortfall\b", r"\bunderperform\b", r"\bdownside\b",
        r"\bconcern\b", r"\brisk(?:s|y)?\b", r"\bthreat\b",
        r"\btumble[ds]?\b", r"\bdiv(?:e|ed|ing)\b", r"\bsink(?:s|ing|ed)?\b",
        r"\bsank\b", r"\bfree.?fall\b", r"\bslash(?:ed|es|ing)?\b",
        r"\bhalt(?:ed|s|ing)?\b", r"\bfreeze\b", r"\bdefault(?:s|ed)?\b",
        r"\bbankrupt(?:cy)?\b", r"\brecall(?:s|ed)?\b", r"\bscandal\b",
        r"\bprob(?:e|ed|ing)\b", r"\bhammered\b", r"\bbattered\b",
        r"\bglut\b", r"\bdeficit\b", r"\bheadwind[s]?\b",
        r"\bstagnation\b", r"\bretreat(?:ed|ing)?\b",
        r"\bdwindl(?:e|ed|ing)\b", r"\bshrink(?:s|ing|age)?\b",
        r"\bprofit warning\b", r"\bguidance cut\b", r"\bmissed estimates?\b",
    ],
    "neutral": [
        r"\bsteady\b", r"\bunchanged\b", r"\bin.line\b", r"\bexpected\b",
        r"\bstable\b", r"\bflat\b", r"\bmixed\b", r"\bmoderate\b",
        r"\bmaintain(?:s|ed)?\b", r"\breaffirm(?:s|ed)?\b",
        r"\bhold(?:s|ing)?\b", r"\brange.?bound\b",
    ],
}

_COMPILED_STANCE: dict[str, list[re.Pattern]] = {
    stance: [re.compile(p, re.IGNORECASE) for p in patterns]
    for stance, patterns in STANCE_KEYWORDS.items()
}

# ---------------------------------------------------------------------------
# Negation handling (stance only — event nouns rarely need it)
# ---------------------------------------------------------------------------
# A bullish/bearish term inside a 4-token negation window flips polarity:
#   "not strong"        → not bullish
#   "no longer growing" → not bullish
#   "without weakness"  → not bearish
NEGATION_WORDS = {"not", "no", "never", "hardly", "barely", "without", "n't", "neither", "nor"}
NEGATION_WINDOW = 4
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?")


def _is_negated(text: str, match_start: int) -> bool:
    """Return True if a negation word appears within NEGATION_WINDOW tokens before match_start."""
    pre = text[:match_start].lower()
    tokens = _TOKEN_RE.findall(pre)
    return any(t in NEGATION_WORDS for t in tokens[-NEGATION_WINDOW:])


# ---------------------------------------------------------------------------
# Keyword Classifiers
# ---------------------------------------------------------------------------

class KeywordEventClassifier:
    """
    Rule-based event classifier using keyword pattern matching.

    Counts keyword hits per category and returns the category with the
    highest normalised hit density.  Returns (prediction, confidence).
    """

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict event type from keyword hits.

        Returns
        -------
        tuple[str, float]
            (predicted_event_type, confidence 0.0–1.0).
            Returns ("Other", 0.0) if no keywords match.
        """
        if not text or not isinstance(text, str):
            return "Other", 0.0

        scores, _ = self._score_with_evidence(text)
        total = sum(scores.values())
        if total == 0:
            return "Market_Movement", 0.0
        best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_cat] / total
        return best_cat, round(confidence, 4)

    def predict_scores(self, text: str) -> dict[str, float]:
        """Return normalised scores for all event categories."""
        scores, _ = self.predict_with_evidence(text)
        return scores

    def predict_with_evidence(
        self, text: str
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """
        Return ``(normalised_scores, matched_terms_per_category)``.

        ``matched_terms_per_category`` maps each category to a list of
        the literal text fragments (lowercased, deduped) that fired.
        Useful for explaining the prediction to the user.
        """
        if not text or not isinstance(text, str):
            return ({cat: 0.0 for cat in EVENT_KEYWORDS},
                    {cat: [] for cat in EVENT_KEYWORDS})

        scores, evidence = self._score_with_evidence(text)
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores, evidence

    @staticmethod
    def _score_with_evidence(
        text: str,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for cat, patterns in _COMPILED_EVENT.items():
            hits = 0
            seen: list[str] = []
            for p in patterns:
                m = p.search(text)
                if m:
                    hits += 1
                    frag = m.group(0).lower()
                    if frag not in seen:
                        seen.append(frag)
            scores[cat] = hits / len(patterns) if patterns else 0.0
            evidence[cat] = seen
        return scores, evidence


class KeywordStanceClassifier:
    """
    Rule-based stance classifier using keyword pattern matching.

    Counts bullish vs bearish keyword hits to determine stance, with a
    short negation window so ``"not strong"`` doesn't fire bullish.
    """

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict stance from keyword hits.

        Returns
        -------
        tuple[str, float]
            (predicted_stance, confidence 0.0–1.0).
            Returns ("neutral", 0.0) if no keywords match.
        """
        if not text or not isinstance(text, str):
            return "neutral", 0.0

        scores, _ = self._score_with_evidence(text)
        total = sum(scores.values())
        if total == 0:
            return "neutral", 0.0
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best] / total
        return best, round(confidence, 4)

    def predict_scores(self, text: str) -> dict[str, float]:
        """Return normalised scores for all stance classes."""
        scores, _ = self.predict_with_evidence(text)
        return scores

    def predict_with_evidence(
        self, text: str
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Return ``(normalised_scores, matched_terms_per_stance)``."""
        if not text or not isinstance(text, str):
            return ({s: 0.0 for s in STANCE_KEYWORDS},
                    {s: [] for s in STANCE_KEYWORDS})

        scores, evidence = self._score_with_evidence(text)
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return scores, evidence

    @staticmethod
    def _score_with_evidence(
        text: str,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for stance, patterns in _COMPILED_STANCE.items():
            hits = 0
            seen: list[str] = []
            for p in patterns:
                # Find the first non-negated match for this pattern.
                fired = False
                for m in p.finditer(text):
                    if _is_negated(text, m.start()):
                        continue
                    fired = True
                    frag = m.group(0).lower()
                    if frag not in seen:
                        seen.append(frag)
                    break
                if fired:
                    hits += 1
            scores[stance] = hits / len(patterns) if patterns else 0.0
            evidence[stance] = seen
        return scores, evidence


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    event_clf = KeywordEventClassifier()
    stance_clf = KeywordStanceClassifier()

    tests = [
        "Apple reported strong revenue growth in its quarterly earnings",
        "CEO announces resignation amid board restructuring",
        "Government introduces new regulations on data privacy",
        "Two major firms announce merger deal worth billions",
        "Stock plunges after disappointing earnings report",
        "Fed holds interest rates steady as widely expected",
    ]

    print("Keyword Classifier Tests:")
    print("-" * 70)
    for t in tests:
        ev, ev_conf = event_clf.predict(t)
        st, st_conf = stance_clf.predict(t)
        print(f"  [{ev:35s} {ev_conf:.2f}]  [{st:8s} {st_conf:.2f}]  {t[:60]}")
