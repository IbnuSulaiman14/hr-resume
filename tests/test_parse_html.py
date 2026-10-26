from datetime import date

import pytest

from src.ingest.parse_html import (
    education_level,
    employment_gaps,
    estimate_years_experience,
    job_timeline,
    extract_skills,
    normalize_section,
    parse_date,
    parse_resume,
    reference_date,
)

# Potongan HTML dengan struktur yang sama seperti Resume_html di dataset
SAMPLE_HTML = """
<div id="document">
  <div class="section firstsection" id="SECTION_NAME1"><div class="paragraph">
    <div class="name"><span class="field"> </span><span class="field"> EXECUTIVE CHEF

CHEF</span></div></div></div>
  <div class="section" id="SECTION_SUMM2">
    <div class="heading"><div class="sectiontitle"> Professional Summary</div></div>
    <div class="paragraph"><p>Chef with <b><u>catering</u></b> background.</p></div></div>
  <div class="section" id="SECTION_HILT3">
    <div class="heading"><div class="sectiontitle"> Highlights</div></div>
    <div class="paragraph"><p>Menu development</p><p>ServSafe certified, Inventory control</p></div></div>
  <div class="section" id="SECTION_EXPR4">
    <div class="heading"><div class="sectiontitle"> Work History</div></div>
    <div class="paragraph"><span class="paddedline"><span class="jobtitle"> Executive Chef</span>
      <span class="datesWrapper"><span class="jobdates"> 01/2015</span><span> to </span>
      <span class="jobdates"> Current</span></span></span>
      <span class="jobline"><ul><li>Led banquet kitchen.</li></ul></span></div>
    <div class="paragraph"><span class="paddedline"><span class="jobtitle"> Line Cook</span>
      <span class="datesWrapper"><span class="jobdates"> Jun 2010</span><span> to </span>
      <span class="jobdates"> Dec 2014</span></span></span></div></div>
  <div class="section" id="SECTION_EDUC5">
    <div class="heading"><div class="sectiontitle"> Education and Training</div></div>
    <div class="paragraph"><span class="degree"> Associate of Applied Science</span> : Culinary Arts
      <span class="jobdates"> 2009</span></div></div>
  <div class="section" id="SECTION_SKLL6">
    <div class="paragraph"><p>Cooking, Catering, clients</p></div></div>
</div>
"""


@pytest.mark.parametrize(
    "title, code, expected",
    [
        ("Work History", None, "experience"),
        ("Professional Experience", "EXPR", "experience"),
        ("Professional Summary", None, "summary"),
        ("Executive Profile", None, "summary"),
        ("Summary of Skills", None, "skills"),
        ("Highlights", "HILT", "skills"),
        ("Education and Training", None, "education"),
        ("Licenses", None, "certifications"),
        ("Accomplishments", "ACCM", "other"),
        ("", "SKLL", "skills"),
        ("Professional Development", "EDUC", "education"),
    ],
)
def test_normalize_section(title, code, expected):
    assert normalize_section(title, code) == expected


def test_parse_resume_sections_and_metadata():
    parsed = parse_resume("123", "CHEF", SAMPLE_HTML, "teks")

    assert list(parsed.sections) == ["summary", "skills", "experience", "education"]
    assert parsed.headline == "EXECUTIVE CHEF"
    # Teks dalam <b><u> tidak terpecah ke baris baru
    assert "Chef with catering background." in parsed.sections["summary"]
    # Highlights dan Skills (tanpa judul, kode SKLL) digabung ke satu seksi
    assert "Menu development" in parsed.sections["skills"] and "Cooking" in parsed.sections["skills"]
    assert parsed.job_titles == ["Executive Chef", "Line Cook"]
    assert parsed.job_dates == [("01/2015", "Current"), ("Jun 2010", "Dec 2014")]
    assert parsed.degrees == ["Associate of Applied Science"]
    assert {"menu development", "servsafe certified", "inventory control", "cooking"} <= set(parsed.skills)


def test_parse_resume_pairs_title_with_dates_per_job():
    parsed = parse_resume("123", "CHEF", SAMPLE_HTML, "teks")

    assert parsed.jobs == [
        {"title": "Executive Chef", "start": "01/2015", "end": "Current"},
        {"title": "Line Cook", "start": "Jun 2010", "end": "Dec 2014"},
    ]


def test_job_timeline_newest_first_with_months():
    jobs = [
        {"title": "Line Cook", "start": "06/2010", "end": "12/2014"},
        {"title": "Executive Chef", "start": "01/2015", "end": "Current"},
        {"title": "Volunteer", "start": "", "end": ""},
    ]
    timeline = job_timeline(jobs, today=date(2017, 1, 1))

    assert [e["title"] for e in timeline] == ["Executive Chef", "Line Cook", "Volunteer"]
    assert [e["months"] for e in timeline] == [24, 54, None]


def test_employment_gaps_after_merging_overlaps():
    dates = [("01/2010", "06/2012"), ("01/2011", "12/2012"), ("01/2014", "01/2015"), ("03/2015", "Current")]

    gaps = employment_gaps(dates, today=date(2016, 1, 1))

    # 12/2012 sampai 01/2014 = 13 bulan; jeda 2 bulan di 2015 di bawah ambang 6 bulan
    assert gaps == [{"from": "2012-12", "to": "2014-01", "months": 13}]


def test_parse_resume_falls_back_to_plain_text():
    parsed = parse_resume("9", "HR", "<div>tanpa seksi</div>", "Resume text  only")

    assert parsed.sections == {"other": "Resume text only"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("09/2016", (2016, 9)),
        ("3/2019", (2019, 3)),
        ("Dec 2013", (2013, 12)),
        ("Dec. 2011", (2011, 12)),
        ("September, 2012", (2012, 9)),
        ("May 21, 2010", (2010, 5)),
        ("2010-12-27", (2010, 12)),
        ("9/2/2015", (2015, 9)),
        ("5/96", (1996, 5)),
        ("August-05", (2005, 8)),
        ("Fall 2014", (2014, 9)),
        ("2015", (2015, None)),
        ("Current", "current"),
        ("present", "current"),
        ("", None),
        ("13/2015", None),
        ("63.16", None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_years_with_current_uses_given_date():
    assert estimate_years_experience([("01/2015", "Current")], today=date(2018, 1, 1)) == 3.0


def test_years_current_defaults_to_latest_date_plus_one_year():
    # Tanggal eksplisit terbaru 01/2015, jadi Current dihitung sampai 01/2016
    assert estimate_years_experience([("01/2015", "Current")]) == 1.0
    assert estimate_years_experience([("01/2010", "01/2015"), ("01/2015", "Current")]) == 6.0


def test_reference_date_uses_latest_year_in_text():
    # Teks menyebut 2014, lebih baru dari tanggal kerja 09/2009, jadi acuannya pertengahan 2015
    assert reference_date([("09/2009", "Current")], "Adopted Scrum since 2014") == date(2015, 6, 1)
    assert reference_date([("09/2009", "Current")], "") == date(2010, 9, 1)
    # Tahun setelah snapshot (mis. rencana lulus) dibatasi
    assert reference_date([("01/2020", "Current")], "Expected 2024") == date(2021, 1, 1)
    assert reference_date([], "") is None


def test_years_overlapping_ranges_counted_once():
    dates = [("01/2010", "01/2014"), ("01/2012", "01/2016")]
    assert estimate_years_experience(dates) == 6.0


def test_years_empty_or_unreadable_returns_none():
    assert estimate_years_experience([]) is None
    assert estimate_years_experience([("", ""), ("tidak", "terbaca")]) is None


def test_years_skips_reversed_range():
    assert estimate_years_experience([("01/2016", "01/2014"), ("01/2010", "01/2011")]) == 1.0


def test_years_only_year_counts_full_years():
    assert estimate_years_experience([("2012", "2014")]) == 3.0


@pytest.mark.parametrize(
    "degrees, expected",
    [
        (["Bachelor of Science"], "bachelor"),
        (["B.S."], "bachelor"),
        (["MBA"], "master"),
        (["Bachelors of Art", "Masters"], "master"),
        (["Ph.D."], "doctorate"),
        (["Associate of Applied Science"], "associate"),
        (["High School Diploma"], "high_school"),
        (["Some College (No Degree)"], "high_school"),
        # "as" huruf kecil adalah kata biasa, bukan gelar Associate of Science
        (["Training course such as Excel"], "unknown"),
        (["Technical Certificate"], "unknown"),
        ([], "unknown"),
    ],
)
def test_education_level(degrees, expected):
    assert education_level(degrees) == expected


def test_extract_skills_drops_contact_lines():
    text = "Excel, SQL\nReferences: Jane Doe (*) 910-555-1234; John Roe john@example.com"
    assert extract_skills(text) == ["excel", "sql"]


def test_extract_skills_drops_sentences_and_generic_words():
    text = "Excel, SQL; Project Management\nLed a team of ten people across three regional offices\nclients"
    assert extract_skills(text) == ["excel", "sql", "project management"]
