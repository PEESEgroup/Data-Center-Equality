#!/usr/bin/env python3
"""Independent re-extraction of PwC (2025) state tables from pwc.pdf.

Outputs:
  pwc_state_data_REEXTRACT.csv   - per-state appendix tables A-3..A-53 (2022, 2023)
  pwc_summary_tables.csv         - summary tables A-1a/A-1b/A-2a/A-2b (thousands / $bn)
"""
import re, sys, os, json

# Release-set path rule: no absolute paths.  Resolved from this file's location.
EMP = os.path.dirname(os.path.abspath(__file__))          # employment
OUT = os.path.join(EMP, "../results/r6_employment")                      # generated results
TXT = os.path.join(OUT, "pwc_layout.txt")

# The PwC (2025) report is a third-party PDF and is NOT redistributed with this
# repository.  Its `pdftotext -layout` rendering IS shipped, as
# pwc_layout.txt, so this script re-runs without the PDF; the PDF is
# needed only to regenerate that rendering from scratch.  Point PWC_PDF at a
# local copy to do so.
PDF = os.environ.get("PWC_PDF")
if not os.path.exists(TXT):
    if not PDF:
        sys.exit(f"missing {TXT}. Set PWC_PDF to a local copy of the PwC (2025) "
                 "report to regenerate it with `pdftotext -layout`.")
    import subprocess
    subprocess.run(["pdftotext", "-layout", PDF, TXT], check=True)

lines = open(TXT, encoding="utf-8").read().split("\n")

# ---- locate page breaks so we can report page numbers -------------------
# pdftotext emits \f between pages; we re-read with page markers
raw = open(TXT, encoding="utf-8").read()
pages = raw.split("\f")
# map line index -> pdf page number (1-based)
# pdftotext puts \f at the START of each new page's text; the printed folio
# ("PwC | ... 29") is the LAST line of that same page.  Walk the lines and bump
# the page counter on every \f.  Verified: Table A-1a -> folio 29,
# Table A-47 (Utah) -> folio 79, Table A-50 (Washington) -> folio 83.
line_page = {}
_pg = 1
for i, l in enumerate(lines):
    if l.startswith("\f"):
        _pg += 1
    line_page[i] = _pg

NUM = re.compile(r"\(?\$?-?[\d,]+\)?%?")

def nums(s):
    """all plain numbers on a line (strip $ , % and trailing pct)"""
    out = []
    for tok in re.findall(r"\$?-?[\d][\d,]*", s):
        t = tok.replace("$", "").replace(",", "")
        out.append(int(t))
    return out

def pcts(s):
    return re.findall(r"-?\d+%", s)

# ---------------- per-state appendix tables ------------------------------
hdr = re.compile(r"^\s*Table A-(\d+[a-z]?):\s*The economic contribution of the data center industry in (.+?),?\s*$")
starts = []
for i, l in enumerate(lines):
    m = hdr.match(l)
    if m:
        starts.append((i, m.group(1), m.group(2).strip()))

# state name may continue on next line ("in the District" / "of Columbia,")
recs = []
for idx, (i, tno, sname) in enumerate(starts):
    yr = lines[i + 1].strip()
    if not re.match(r"^\d{4}-\d{4}$", yr):
        # state name wrapped onto the next line, possibly with the years
        nxt = lines[i + 1].strip()
        m2 = re.match(r"^(.*?),?\s*(\d{4}-\d{4})$", nxt)
        if m2:
            sname = (sname + " " + m2.group(1).rstrip(",")).strip()
            yr = m2.group(2)
        else:
            sname = (sname + " " + nxt.rstrip(",")).strip()
            yr = lines[i + 2].strip()
    sname = re.sub(r"^the\s+", "", sname)
    end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
    recs.append(dict(tno=tno, state=sname, years=yr, lo=i, hi=end,
                     page=line_page.get(i)))

SECT = {
    "Employment (jobs)": "emp",
    "Labor Income ($millions)": "li",
    "GDP ($millions)": "gdp",
    "Total State and Local Tax Contribution": "tax",
}

ROWLAB = [
    ("direct_contribution",      re.compile(r"^\s*Direct contribution")),
    ("indirect_induced",         re.compile(r"^\s*Indirect and induced")),
    ("operational",              re.compile(r"^\s*Operational")),
    ("capital_spending",         re.compile(r"^\s*Capital Spending")),
    ("total_no_spillover",       re.compile(r"^\s*Total contribution without")),
    ("cross_state_spillover",    re.compile(r"^\s*Cross-state spillover")),
    ("total_with_spillover",     re.compile(r"^\s*Total contribution with")),
    ("tax_no_spillover",         re.compile(r"^\s*Without cross-state spillover")),
    ("tax_with_spillover",       re.compile(r"^\s*With cross-state spillover")),
]

def parse_block(rec):
    """returns {(sect,row): [values...]} for a 2022-2023 table"""
    out, sect = {}, None
    blk = lines[rec["lo"]:rec["hi"]]
    j = 0
    while j < len(blk):
        l = blk[j]
        s = l.strip()
        hit = None
        for k, v in SECT.items():
            if s.startswith(k):
                hit = v
        if hit:
            sect = hit
            j += 1
            continue
        if sect:
            for name, pat in ROWLAB:
                if pat.match(l):
                    v = nums(re.sub(r"-?\d+%\s*$", "", l))
                    look = j
                    # label wrapped: numbers on a following line
                    while len(v) == 0 and look + 1 < len(blk) and look - j < 3:
                        look += 1
                        v = nums(re.sub(r"-?\d+%\s*$", "", blk[look]))
                    out[(sect, name)] = v
                    j = look
                    break
        j += 1
    return out

parsed = {}
for rec in recs:
    if rec["years"] != "2022-2023":
        continue
    d = parse_block(rec)
    parsed[rec["state"]] = dict(rec=rec, data=d)

print("per-state 2022-2023 tables parsed:", len(parsed))

order = ["emp", "li", "gdp"]
rows = ["direct_contribution", "indirect_induced", "operational", "capital_spending",
        "total_no_spillover", "cross_state_spillover", "total_with_spillover"]

bad = []
out_rows = []
for st, o in parsed.items():
    d, r = o["data"], o["rec"]
    rec = {"state": st, "table": "A-" + r["tno"], "pdf_page": r["page"]}
    for sec in order:
        for row in rows:
            v = d.get((sec, row))
            if v is None or len(v) != 2:
                bad.append((st, sec, row, v))
                v = [None, None]
            rec[f"{sec}_{row}_2022"] = v[0]
            rec[f"{sec}_{row}_2023"] = v[1]
    for row in ["tax_no_spillover", "tax_with_spillover"]:
        v = d.get(("tax", row))
        if v is None or len(v) != 2:
            bad.append((st, "tax", row, v))
            v = [None, None]
        rec[f"{row}_2022"] = v[0]
        rec[f"{row}_2023"] = v[1]
    out_rows.append(rec)

if bad:
    print("PARSE PROBLEMS:")
    for b in bad:
        print("   ", b)

import pandas as pd
df = pd.DataFrame(out_rows).sort_values("state").reset_index(drop=True)
df.to_csv(os.path.join(OUT, "pwc_state_data_REEXTRACT.csv"), index=False)
print("wrote pwc_state_data_REEXTRACT.csv", df.shape)

# ---------------- summary tables A-1a / A-1b / A-2a / A-2b ---------------
sumhdr = re.compile(r"^\s*Table A-(1a|1b|2a|2b)\.")
sstarts = [(i, sumhdr.match(l).group(1)) for i, l in enumerate(lines) if sumhdr.match(l)]
srows = []
STATES = sorted({r["state"] for r in recs} | {"District of Columbia"})
namefix = {"the District of Columbia": "District of Columbia"}
statenames = sorted({namefix.get(s, s) for s in STATES})

for k, (i, tag) in enumerate(sstarts):
    end = sstarts[k + 1][0] if k + 1 < len(sstarts) else i + 70
    pg = line_page.get(i)
    for l in lines[i:end]:
        s = l.strip()
        for st in statenames:
            if s.startswith(st + " ") or s == st:
                rest = s[len(st):]
                v = re.findall(r"\$?-?[\d][\d,]*\.?\d*", rest)
                v = [float(x.replace("$", "").replace(",", "")) for x in v]
                if len(v) == 6:
                    srows.append(dict(table="A-" + tag, pdf_page=pg, state=st,
                                      emp_direct=v[0], emp_total=v[1],
                                      li_direct=v[2], li_total=v[3],
                                      gdp_direct=v[4], gdp_total=v[5]))
                break
sdf = pd.DataFrame(srows)
sdf.to_csv(os.path.join(OUT, "pwc_summary_tables.csv"), index=False)
print("wrote pwc_summary_tables.csv", sdf.shape)
print(sdf.groupby("table").size())
