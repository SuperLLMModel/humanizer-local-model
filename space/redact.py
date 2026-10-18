"""Strip obvious personal details from text before it is stored (usage log, text_v1/).

redact(text) -> (scrubbed, counts): each match is replaced by a placeholder such as [EMAIL] or [PHONE],
and counts says how many of each were replaced. Regex only: names, addresses and anything written out
in words are not caught, so the page never promises that everything personal is removed.

Placeholders: [EMAIL] [URL] [SECRET] [IP] [ID] [CARD] [SSN] [PHONE] [HANDLE].
Order matters: e-mail before URL (a bare domain inside an address), keys and IDs before phones and cards
(their digit runs would otherwise be split), phones before cards (a 13-digit +86 mobile can pass the
Luhn check), long digit runs last.
"""
from __future__ import annotations

import re
from collections import Counter

# Chinese text has no spaces, so "电话13812345678" must still match: digit and letter lookarounds
# instead of \b (Python's \b treats a Chinese character as a word character).
_ND = r"(?<!\d)"
_NDA = r"(?<![\dA-Za-z])"
_ED = r"(?!\d)"
_EDA = r"(?![\dA-Za-z])"

_URLCH = r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
_TLDS = ("com|net|org|edu|gov|mil|int|io|ai|co|cn|uk|us|me|app|dev|info|biz|xyz|top|cc|tv|ly|gl|gg|so|jp|de|fr|"
         "hk|tw|sg|ca|au|in|ru|kr|nl|es|it|br|ch|se|no|nz|eu|site|online|link|page|blog|shop|store|tech")

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+\s?[@＠]\s?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# what is left of an address after the TLD got split off ("doe@example."): still an address
EMAIL_LOOSE = re.compile(r"[A-Za-z0-9._%+\-]*[A-Za-z0-9][@＠][A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*")
URL = re.compile(
    r"(?i:(?:https?|ftp)://|www\.)" + _URLCH + r"+"
    # bare domain: lowercase TLD only (so "finished.So we" is not a domain), at least one label before it
    + r"|(?<![A-Za-z0-9_@.\-/])(?=[A-Za-z0-9\-]*[A-Za-z])(?:[A-Za-z0-9\-]+\.)+(?:" + _TLDS + r")(?![A-Za-z0-9\-])"
    + r"(?:/" + _URLCH + r"*)?")
# TLDs that are also short words or abbreviations: a bare "x.in" / "a.m.in" / "e.g.so" needs a path to count
_WORDY_TLDS = {"in", "it", "no", "me", "so", "us", "co", "de", "es", "se", "ch", "nl", "cc", "tv", "ly", "gl", "gg"}
SECRET = re.compile(
    r"(?<![A-Za-z0-9_\-])(?:sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[abposr]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})")
UUID = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}(?![0-9A-Fa-f])")
_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4 = re.compile(r"(?<![\d.])" + _OCT + r"(?:\." + _OCT + r"){3}(?!\d)(?!\.\d)")
IPV6 = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])")
CN_ID = re.compile(_NDA + r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]" + _EDA)
CARD = re.compile(_ND + r"\d(?:[ \-]?\d){12,18}" + _ED)
SSN = re.compile(_ND + r"(?!000|666|9\d\d)\d{3}([\- ])(?!00)\d{2}\1(?!0000)\d{4}" + _ED)
SSN_KW = re.compile(r"(?i)(\b(?:SSN|social security(?: number| no\.?)?)\s*(?:#|:|：|is)?\s*)\d{9}" + _ED)
PHONE_CN_MOBILE = re.compile(_ND + r"(?:(?:\+|00)?86[\s\-]?)?1[3-9]\d(?:[\s\-]?\d{4}){2}" + _ED)
PHONE_CN_LAND = re.compile(_ND + r"(?:\(0\d{2,3}\)\s?|0\d{2,3}[\s\-])\d{3,4}[\s\-]?\d{4}" + _ED)
PHONE_US = re.compile(r"(?<![\dA-Za-z+])(?:\+?1[\s.\-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.\-]?[2-9]\d{2}[\s.\-]?\d{4}" + _ED)
PHONE_INTL = re.compile(r"(?<![\dA-Za-z])\+\d{1,3}(?:[\s.\-]?\(?\d{1,5}\)?){2,6}" + _ED)
_KW = (r"account(?: number| no\.?| #| id)?|acct(?: no\.?| #)?|a/c|user\s?name|user\s?id|login(?: id)?|"
       r"member(?:ship)?\s?(?:id|no\.?|number|#)|customer\s?(?:id|no\.?|number|#)|order\s?(?:id|no\.?|number|#)|"
       r"employee\s?(?:id|no\.?|number)|student\s?(?:id|no\.?|number)|passport(?: no\.?| number)?|"
       r"driver'?s licen[cs]e(?: no\.?| number)?|licen[cs]e (?:no\.?|number)|routing (?:no\.?|number)|iban|swift(?: code)?|"
       r"tracking (?:no\.?|number|#)|case (?:no\.?|number|#)|ticket (?:no\.?|number|#)|policy (?:no\.?|number)|"
       r"账号|账户|帐号|帐户|卡号|银行卡号?|身份证号?码?|护照号?码?|用户名|用户\s?ID|会员号|订单号|学号|工号|员工号|"
       r"QQ号?|微信号?|WeChat(?: ID)?|支付宝(?:账号)?|手机尾号")
CONTEXT_ID = re.compile(r"(?i)(?<![A-Za-z])(" + _KW + r")(\s*(?:is|was|:|：|#|为|是)?\s*)"
                        r"([A-Za-z0-9][A-Za-z0-9_\-.]*[A-Za-z0-9]|[A-Za-z0-9])(?![A-Za-z0-9_\-])")
LONG_DIGITS = re.compile(r"(?<![\d,.])\d{10,}(?![\d,.]?\d)")
# no hyphen: "CRISPR-Cas9-mediated" is a word, not a token (hyphenated keys are caught by SECRET / UUID)
TOKEN = re.compile(r"(?<![A-Za-z0-9_\-\[])(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Za-z])[A-Za-z0-9_]{20,}(?![A-Za-z0-9_\-\]])")
HANDLE = re.compile(r"(?<![A-Za-z0-9_.@\[])@[A-Za-z0-9_](?:[A-Za-z0-9_.]{0,29}[A-Za-z0-9_])?")

_TRAIL = ".,;:!?)'\"]"


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _sub_url(m: re.Match) -> str:
    s = m.group(0)
    if "://" not in s and not s.lower().startswith("www."):
        host, _, path = s.partition("/")
        labels = host.split(".")
        if len(labels[-2]) < 2 and len(labels) == 2 or all(len(x) == 1 for x in labels[:-1]):
            return s        # "a.m.in", "x.so": abbreviations, not hosts
        if labels[-1] in _WORDY_TLDS and not path:
            return s
    tail = ""
    while s and s[-1] in _TRAIL:
        # keep a closing bracket that belongs to the URL, e.g. ...wiki/Foo_(bar) or ?e=[EMAIL]
        if s[-1] == ")" and s.count("(") >= s.count(")") or s[-1] == "]" and s.count("[") >= s.count("]"):
            break
        tail = s[-1] + tail
        s = s[:-1]
    return "[URL]" + tail


def _sub_ipv6(m: re.Match) -> str:
    s = m.group(0)
    groups = [g for g in s.split(":") if g]
    # real addresses: 8 groups, or "::" with at least 3 groups and a digit (not C++ "a::b" or a time "10:30:00")
    if (s.count(":") == 7 and len(groups) == 8) or ("::" in s and len(groups) >= 3 and re.search(r"\d", s)
                                                       and any(re.search(r"[A-Fa-f]", g) or len(g) == 4 for g in groups)):
        return "[IP]"
    return s


def _sub_card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)


def _sub_intl(m: re.Match) -> str:
    n = len(re.sub(r"\D", "", m.group(0)))
    return "[PHONE]" if 8 <= n <= 15 else m.group(0)


def _sub_context(m: re.Match) -> str:
    kw, sep, val = m.group(1), m.group(2), m.group(3)
    if val.startswith("[") or val.upper() in {"IS", "WAS", "NUMBER", "NO"}:
        return m.group(0)
    strong_sep = any(c in sep for c in ":：#为是")
    if not strong_sep and not re.search(r"\d", val):
        return m.group(0)       # "account is overdue", "order history": no digit, no colon -> not an ID
    if len(val) < 4 and not re.search(r"\d", val):
        return m.group(0)
    return kw + sep + "[ID]"


# (placeholder, pattern, replacement): replacement is a string or a function returning the new text
_RULES = [
    ("EMAIL", EMAIL, "[EMAIL]"),
    ("EMAIL", EMAIL_LOOSE, "[EMAIL]"),
    ("URL", URL, _sub_url),
    ("SECRET", SECRET, "[SECRET]"),
    ("ID", UUID, "[ID]"),
    ("IP", IPV4, "[IP]"),
    ("IP", IPV6, _sub_ipv6),
    ("ID", CN_ID, "[ID]"),
    ("SSN", SSN, "[SSN]"),
    ("SSN", SSN_KW, lambda m: m.group(1) + "[SSN]"),
    ("PHONE", PHONE_CN_MOBILE, "[PHONE]"),
    ("PHONE", PHONE_INTL, _sub_intl),
    ("PHONE", PHONE_US, "[PHONE]"),
    ("PHONE", PHONE_CN_LAND, "[PHONE]"),
    ("CARD", CARD, _sub_card),
    ("ID", CONTEXT_ID, _sub_context),
    ("ID", LONG_DIGITS, "[ID]"),
    ("ID", TOKEN, "[ID]"),
    ("HANDLE", HANDLE, "[HANDLE]"),
]


def redact(text: str) -> tuple[str, dict]:
    """(text with personal details replaced by placeholders, {placeholder: count})."""
    if not text:
        return text or "", {}
    counts: Counter = Counter()
    for name, pat, repl in _RULES:
        tag = f"[{name}]"

        def fn(m, repl=repl, tag=tag):
            new = repl(m) if callable(repl) else repl
            if new != m.group(0):
                counts[name] += new.count(tag) or 1
            return new
        text = pat.sub(fn, text)
    return text, dict(counts)


def scrub(text: str) -> str:
    return redact(text)[0]
