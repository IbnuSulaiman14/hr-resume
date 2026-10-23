"""Parse kolom Resume_html menjadi seksi dan metadata. Tahap B1.1 sampai B1.3.

Fungsi di sini murni (tanpa jaringan) supaya mudah diuji.
"""
import re
from dataclasses import dataclass, field
from datetime import date

from bs4 import BeautifulSoup, NavigableString, Tag


@dataclass
class ParsedResume:
    resume_id: str
    category: str
    sections: dict[str, str] = field(default_factory=dict)  # nama seksi ternormalisasi -> teks
    headline: str = ""  # judul di bagian atas CV (dataset menaruh jabatan di kolom nama)
    job_titles: list[str] = field(default_factory=list)
    job_dates: list[tuple[str, str]] = field(default_factory=list)  # (mulai, selesai) teks mentah
    degrees: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)


# Urutan penting: "summary of skills" harus jadi skills, "education and certifications" jadi education.
_TITLE_RULES = [
    ("experience", r"experience|work history|employment|career history"),
    ("education", r"education|academic"),
    ("certifications", r"certif|licen|credential"),
    ("skills", r"skill|qualification|highlight|expertise|competenc|strength|proficienc|technical"),
    ("summary", r"summary|profile|overview|objective|career focus"),
]

# Kode di id elemen (SECTION_EXPR...) dipakai bila judul kosong atau tidak cocok aturan mana pun.
_CODE_TO_SECTION = {
    "SUMM": "summary", "OBJC": "summary",
    "SKLL": "skills", "HILT": "skills", "TSKL": "skills",
    "EXPR": "experience", "WRKH": "experience",
    "EDUC": "education",
    "CERT": "certifications",
}

_BLOCK_TAGS = {"p", "li", "div", "br", "tr", "td", "table", "ul", "ol", "h1", "h2", "h3", "h4"}
_LINE_CLASSES = {"paddedline", "jobline", "jobtitle"}


def normalize_section(title: str, code: str | None = None) -> str:
    """Petakan judul seksi mentah ("Work History", "Professional Summary") ke salah satu config.SECTIONS.
    `code` adalah kode empat huruf dari id elemen (EXPR, SKLL, ...) sebagai cadangan."""
    lowered = (title or "").lower()
    for name, pattern in _TITLE_RULES:
        if re.search(pattern, lowered):
            return name
    return _CODE_TO_SECTION.get(code or "", "other")


def _clean_lines(text: str) -> str:
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.replace("\xa0", " ").splitlines())
    return "\n".join(line for line in lines if line)


def _block_text(node: Tag) -> str:
    """Teks dengan baris baru hanya di batas elemen blok, supaya kata dalam <b> atau <u> tidak terpotong."""
    parts: list[str] = []

    def walk(el) -> None:
        for child in el.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                walk(child)
                classes = set(child.get("class") or [])
                if child.name in _BLOCK_TAGS or classes & _LINE_CLASSES:
                    parts.append("\n")

    walk(node)
    return _clean_lines("".join(parts))


def _first_line(text: str) -> str:
    return _clean_lines(text).split("\n")[0] if text.strip() else ""


def _section_code(section: Tag) -> str | None:
    match = re.match(r"SECTION_([A-Z]{4})", section.get("id", ""))
    return match.group(1) if match else None


def _job_dates(paragraph: Tag) -> tuple[str, str] | None:
    """Pasangan (mulai, selesai) dari satu entri pekerjaan. Teks penghubung "to" dan span kosong dibuang."""
    values = [d.get_text(" ", strip=True) for d in paragraph.select(".jobdates")]
    values = [v for v in values if v and v.lower() != "to"]
    if len(values) >= 2:
        return values[0], values[-1]
    if len(values) == 1 and " to " in values[0]:
        start, _, end = values[0].partition(" to ")
        return start.strip(), end.strip()
    return None


def parse_resume(resume_id: str, category: str, html: str, text: str) -> ParsedResume:
    """Ambil seksi, jabatan, tanggal, gelar, dan skill dari HTML. Bila tidak ada seksi terparse, kembalikan
    satu seksi "other" berisi `text` (cadangan)."""
    parsed = ParsedResume(resume_id=str(resume_id), category=category)
    soup = BeautifulSoup(html or "", "lxml")

    for section in soup.select("div.section"):
        code = _section_code(section)
        title_el = section.select_one(".sectiontitle")
        if code == "NAME" or (title_el is None and section.select_one(".name")):
            parsed.headline = parsed.headline or _first_line(section.get_text("\n"))
            continue

        title = title_el.get_text(" ", strip=True) if title_el else ""
        name = normalize_section(title, code)
        heading = section.select_one(".heading")
        if heading:
            heading.extract()
        body = _block_text(section)
        if not body:
            continue
        # Seksi dengan nama sama (mis. Skills dan Highlights) digabung
        parsed.sections[name] = f"{parsed.sections[name]}\n{body}" if name in parsed.sections else body

        if name == "experience":
            for paragraph in section.select("div.paragraph"):
                for title_span in paragraph.select(".jobtitle"):
                    job = _first_line(title_span.get_text("\n"))
                    if job and job not in parsed.job_titles:
                        parsed.job_titles.append(job)
                pair = _job_dates(paragraph)
                if pair:
                    parsed.job_dates.append(pair)
        if name == "skills":
            parsed.skills.extend(s for s in extract_skills(body) if s not in parsed.skills)

    parsed.degrees = [d for d in (_clean_lines(el.get_text(" ")) for el in soup.select(".degree")) if d]
    if not parsed.degrees and "education" in parsed.sections:
        parsed.degrees = parsed.sections["education"].split("\n")

    if not parsed.sections:
        fallback = _clean_lines(text or "")
        if fallback:
            parsed.sections["other"] = fallback
    return parsed


# ---------------------------------------------------------------------------
# Tanggal dan tahun pengalaman

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}
_CURRENT = {"current", "present", "now", "today", "till date", "to date", "ongoing"}

# Tanggal terbaru yang masuk akal di dataset (lihat docs/PENJELASAN.md, B1).
DATA_SNAPSHOT = date(2021, 12, 1)
MIN_YEAR = 1950


def _expand_year(two_digits: str) -> int:
    value = int(two_digits)
    return 2000 + value if value <= DATA_SNAPSHOT.year % 100 else 1900 + value


def parse_date(raw: str) -> tuple[int, int | None] | str | None:
    """Ubah teks tanggal menjadi (tahun, bulan atau None), "current", atau None bila tidak terbaca."""
    value = (raw or "").strip().lower().rstrip(".")
    if not value:
        return None
    if value in _CURRENT:
        return "current"

    patterns = [
        (r"(\d{4})-(\d{1,2})-\d{1,2}", lambda m: (int(m[1]), int(m[2]))),         # 2010-12-27
        (r"(\d{1,2})/\d{1,2}/(\d{4})", lambda m: (int(m[2]), int(m[1]))),         # 9/2/2015
        (r"(\d{1,2})[/.\- ](\d{4})", lambda m: (int(m[2]), int(m[1]))),           # 09/2016, 12 2014
        (r"(\d{1,2})/(\d{2})", lambda m: (_expand_year(m[2]), int(m[1]))),        # 5/96
        (r"([a-z]+)\.?,?[\s\-]*(\d{1,2},?\s*)?(\d{4})", None),                    # Dec. 2011, May 21, 2010
        (r"([a-z]+)-(\d{2})", None),                                              # August-05
        (r"(\d{4})", lambda m: (int(m[1]), None)),                                # 2015
    ]
    for pattern, build in patterns:
        match = re.fullmatch(pattern, value)
        if not match:
            continue
        if build is not None:
            result = build(match)
        else:
            word = match[1]
            month = _MONTHS.get(word[:3]) or _SEASONS.get(word)
            if month is None:
                continue
            year = int(match[3]) if len(match.groups()) == 3 else _expand_year(match[2])
            result = (year, month)
        year, month = result
        if month is not None and not 1 <= month <= 12:
            return None
        return result
    return None


def _to_months(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def reference_date(job_dates: list[tuple[str, str]], text: str = "") -> date | None:
    """Perkiraan kapan CV ditulis, dipakai sebagai arti "Current".

    CV paling cepat ditulis pada tanggal terbaru yang disebut di dalamnya (tanggal kerja atau tahun
    mana pun di teks, misalnya "Exchange 2013" atau tahun lulus). Ditambah 12 bulan karena CV biasanya
    diperbarui setelah peristiwa terakhirnya, dan dibatasi DATA_SNAPSHOT. Aturan ini dipilih setelah
    dibandingkan dengan kalimat "N+ years" di ringkasan CV (lihat docs/PENJELASAN.md, B1)."""
    candidates = []
    for start, end in job_dates:
        for parsed in (parse_date(start), parse_date(end)):
            if parsed and parsed != "current":
                candidates.append((parsed[0], parsed[1] or 6))
    for year in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", text or ""):
        if int(year) <= DATA_SNAPSHOT.year:
            candidates.append((int(year), 6))
    if not candidates:
        return None
    months = _to_months(*max(candidates)) + 12
    return min(date(months // 12, months % 12 + 1, 1), DATA_SNAPSHOT)


def estimate_years_experience(job_dates: list[tuple[str, str]], today: date | None = None) -> float | None:
    """Perkiraan total tahun pengalaman dari rentang tanggal kerja. Rentang yang bertumpuk dihitung sekali.

    "Current" dihitung sampai `today` (ingest memakai reference_date). Bila `today` kosong, dipakai tanggal
    kerja eksplisit terbaru ditambah satu tahun (maksimal DATA_SNAPSHOT), karena CV ditulis pada tahun
    yang berbeda-beda.
    Bulan akhir tidak ikut dihitung (01/2010 sampai 01/2014 = 4 tahun). Tahun tanpa bulan dianggap
    awal tahun untuk tanggal mulai dan akhir tahun untuk tanggal selesai (2012 sampai 2014 = 3 tahun).
    Kembalikan None bila tidak ada rentang yang bisa dibaca."""
    pairs = [(parse_date(start), parse_date(end)) for start, end in job_dates]
    pairs = [(s, e) for s, e in pairs if s and e and s != "current"]
    if not pairs:
        return None

    if today is None:
        explicit = [d for pair in pairs for d in pair if d != "current"]
        latest_year = max(y for y, _ in explicit)
        latest_month = max((m or 12) for y, m in explicit if y == latest_year)
        latest = date(latest_year + 1, latest_month, 1)
        today = min(latest, DATA_SNAPSHOT)
    today_months = _to_months(today.year, today.month)

    intervals = []
    for start, end in pairs:
        start_m = _to_months(start[0], start[1] or 1)
        if end == "current":
            end_m = today_months
        else:
            end_m = _to_months(end[0], end[1]) if end[1] else _to_months(end[0] + 1, 1)
        if start[0] < MIN_YEAR or end_m < start_m or end_m > today_months + 12:
            continue
        intervals.append((start_m, end_m))
    if not intervals:
        return None

    total = 0
    cur_start, cur_end = None, None
    for start_m, end_m in sorted(intervals):
        if cur_end is None or start_m > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start_m, end_m
        else:
            cur_end = max(cur_end, end_m)
    total += cur_end - cur_start
    return round(total / 12, 1)


# ---------------------------------------------------------------------------
# Pendidikan dan skill

EDUCATION_LEVELS = ["unknown", "high_school", "associate", "bachelor", "master", "doctorate"]

# (level, kata lengkap tanpa peduli huruf besar, singkatan yang harus berhuruf kapital).
# Singkatan dibedakan karena "as", "be", "ma" juga kata biasa di teks gelar yang kotor.
_EDU_RULES = [
    ("doctorate", r"ph ?d|doctor\w*|juris", r"PhD|EdD|PsyD|PharmD|DNP|MD|JD|DBA"),
    ("master", r"master\w*|graduate degree", r"MBA|MS|MSc|MA|MPH|MSN|MFA|LLM|MEd|MEng|MPA|MSW"),
    ("bachelor", r"bachelor\w*|undergraduate degree|baccalaureate", r"BS|BSc|BA|BBA|BFA|BSN|BEng|BE|LLB|BCom|BTech"),
    ("associate", r"associate\w*", r"AA|AS|AAS|AOS"),
    ("high_school", r"high school|ged|diploma|some college|secondary school", r"GED"),
]


def education_level(degrees: list[str]) -> str:
    """Petakan daftar gelar kotor ke: high_school, associate, bachelor, master, doctorate, atau unknown.
    Diambil level tertinggi yang ditemukan."""
    best = "unknown"
    for degree in degrees:
        # Hapus titik supaya "B.S." dan "Ph.D." terbaca sebagai "BS" dan "PhD"
        text = degree.replace(".", "")
        for level, words, abbrevs in _EDU_RULES:
            if re.search(rf"\b({words})\b", text, re.IGNORECASE) or re.search(rf"\b({abbrevs})\b", text):
                if EDUCATION_LEVELS.index(level) > EDUCATION_LEVELS.index(best):
                    best = level
                break
    return best


# Kata umum yang sering muncul di daftar kata kunci Skills dataset ini tetapi bukan nama skill
_NOT_SKILLS = {
    "client", "clients", "policies", "processes", "managing", "personnel", "meetings", "materials",
    "progress", "basic", "director", "content", "market", "delivery", "profit", "enterprise",
    "financial", "skills", "skill", "teacher", "administrative", "academic",
}


# Email atau nomor telepon: baris seperti ini biasanya daftar referensi, bukan skill
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\(?\d{3}\)?[-. ]?\d{3}[-. ]\d{3,4}|\breferences?\b|\(\*\)", re.IGNORECASE)


def extract_skills(skills_text: str) -> list[str]:
    """Pecah teks seksi Skills menjadi daftar skill pendek (huruf kecil, tanpa duplikat).
    Kalimat panjang dibuang karena biasanya deskripsi, bukan nama skill. Baris berisi kontak dibuang
    seluruhnya supaya nama dan telepon referensi tidak ikut ke metadata."""
    skills: list[str] = []
    for line in skills_text.split("\n"):
        if _CONTACT.search(line):
            continue
        for item in re.split(r"[,;•|·:]|\s-\s", line):
            cleaned = re.sub(r"\s+", " ", item.strip(" .-*()[]\"'").lower())
            if not 2 <= len(cleaned) <= 40 or len(cleaned.split()) > 4 or not re.search(r"[a-z]", cleaned):
                continue
            if cleaned in _NOT_SKILLS:
                continue
            if cleaned not in skills:
                skills.append(cleaned)
    return skills
