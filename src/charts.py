"""Grafik Altair dari spesifikasi yang dicatat tool Analyst (src/tools/analyst_tools.py).

Tool menghitung angkanya dengan Pandas; modul ini hanya menggambar. Semua grafik di sini satu seri:
satu warna, tanpa legenda (judul sudah menyebut isinya), angka di ujung batang, dan tooltip saat hover.

Warna sengaja tidak mengikuti tema: st.context.theme bisa salah sesaat setelah pengguna mengganti tema
(streamlit/streamlit#11920). Dua warna di bawah dipilih supaya kontrasnya cukup di latar terang (#ffffff)
maupun gelap (#0e1117): batang 4,42 dan 4,28; angka 4,35 dan 4,34. Judul dan sumbu diwarnai tema Streamlit.
"""
import altair as alt
import pandas as pd

SERIES_COLOR = "#2a78d6"
VALUE_TEXT_COLOR = "#797979"
BAR_THICKNESS = 16
COLUMN_THICKNESS = 24
ROW_HEIGHT = 24
COLUMN_CHART_HEIGHT = 260


def build_chart(spec: dict) -> alt.LayerChart:
    """Batang horizontal untuk peringkat (kategori, skill), kolom vertikal untuk rentang berurutan (tahun)."""
    data = pd.DataFrame(spec["data"])
    order = list(data["label"])  # urutan dari tool dipertahankan: peringkat atau rentang tahun
    tooltip = [
        alt.Tooltip("label:N", title=spec["label_title"]),
        alt.Tooltip("value:Q", title=spec["value_title"], format=","),
    ]
    if "detail" in data:
        tooltip.append(alt.Tooltip("detail:N", title="Keterangan"))

    value_axis = alt.Axis(title=spec["value_title"], format=",d", tickMinStep=1)
    # Ruang di ujung sumbu supaya angka pada batang terpanjang tidak terpotong
    value_scale = alt.Scale(domainMax=max(float(data["value"].max()), 1.0) * 1.08, nice=True)
    if spec.get("orientation") == "horizontal":
        label = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False))
        value = alt.X("value:Q", axis=value_axis, scale=value_scale)
        bar = {"size": BAR_THICKNESS}
        text = {"align": "left", "baseline": "middle", "dx": 4}
        height = ROW_HEIGHT * len(data) + 40
    else:
        label = alt.X("label:N", sort=order, title=spec["label_title"], axis=alt.Axis(labelAngle=0, ticks=False))
        value = alt.Y("value:Q", axis=value_axis, scale=value_scale)
        bar = {"size": COLUMN_THICKNESS}
        text = {"align": "center", "baseline": "bottom", "dy": -4}
        height = COLUMN_CHART_HEIGHT

    base = alt.Chart(data).encode(label, value, tooltip=tooltip)
    # Ujung data membulat 4px, sisi baseline tetap siku
    bars = base.mark_bar(color=SERIES_COLOR, cornerRadiusEnd=4, **bar)
    values = base.mark_text(color=VALUE_TEXT_COLOR, **text).encode(text=alt.Text("value:Q", format=","))
    return alt.layer(bars, values).properties(
        title=alt.TitleParams(spec["title"], subtitle=spec.get("subtitle", ""), anchor="start"),
        height=height,
    )


def chart_table(spec: dict) -> pd.DataFrame:
    """Data grafik sebagai tabel, untuk yang tidak bisa atau tidak ingin membaca grafik."""
    columns = {"label": spec["label_title"], "value": spec["value_title"], "detail": "Keterangan"}
    data = pd.DataFrame(spec["data"])
    return data[[c for c in columns if c in data]].rename(columns=columns)
