from src.charts import SERIES_COLOR, VALUE_TEXT_COLOR, build_chart, chart_table

RANKING = {
    "title": "Jumlah CV per kategori",
    "subtitle": "3 CV di 2 kategori",
    "orientation": "horizontal",
    "label_title": "Kategori",
    "value_title": "Jumlah CV",
    "data": [{"label": "CHEF", "value": 2, "detail": "66.7%"}, {"label": "HR", "value": 1, "detail": "33.3%"}],
}
YEARS = {
    "title": "Sebaran perkiraan tahun pengalaman",
    "orientation": "vertical",
    "label_title": "Tahun pengalaman (perkiraan)",
    "value_title": "Jumlah CV",
    "data": [{"label": "5-10", "value": 3}, {"label": "10-15", "value": 1}],
}


def test_ranking_is_horizontal_and_keeps_tool_order():
    spec = build_chart(RANKING).to_dict()
    bars = spec["layer"][0]

    assert bars["mark"]["type"] == "bar"
    assert bars["mark"]["color"] == SERIES_COLOR
    assert bars["mark"]["cornerRadiusEnd"] == 4
    assert bars["encoding"]["y"]["sort"] == ["CHEF", "HR"]
    assert spec["layer"][1]["mark"]["type"] == "text"  # angka di ujung batang
    assert spec["layer"][1]["mark"]["color"] == VALUE_TEXT_COLOR
    assert spec["title"]["text"] == "Jumlah CV per kategori"
    assert "legend" not in str(bars["encoding"])


def test_years_are_columns_in_bin_order():
    bars = build_chart(YEARS).to_dict()["layer"][0]

    assert bars["encoding"]["x"]["sort"] == ["5-10", "10-15"]
    assert bars["encoding"]["y"]["scale"]["domainMax"] > 3  # ruang untuk angka di atas kolom tertinggi


def _contrast(a: str, b: str) -> float:
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_colors_readable_on_light_and_dark_background():
    # Warna tidak mengikuti tema, jadi harus cukup kontras di kedua latar (lihat .streamlit/config.toml)
    for background in ["#ffffff", "#0e1117"]:
        assert _contrast(SERIES_COLOR, background) >= 3.0  # batas WCAG untuk objek grafis
        assert _contrast(VALUE_TEXT_COLOR, background) >= 4.3


def test_chart_table_uses_readable_headers():
    table = chart_table(RANKING)

    assert list(table.columns) == ["Kategori", "Jumlah CV", "Keterangan"]
    assert list(chart_table(YEARS).columns) == ["Tahun pengalaman (perkiraan)", "Jumlah CV"]
