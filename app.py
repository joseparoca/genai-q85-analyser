# ──────────────────────────────────────────────────────────────────────────────
#  GenAI Q85 Analyser – Streamlit web‑app
#  Works on Streamlit Community Cloud (free tier)
# ──────────────────────────────────────────────────────────────────────────────
import io
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Q85 Analyser",
    page_icon="🧮",
    layout="wide",
)

st.title("GenAI Survey – Daily Q85 Analysis")
st.markdown(
    """
    Upload the **latest survey export** and the **question‑type mapping**&nbsp;→  
    click&nbsp;**Run analysis** → download a fresh Excel with p‑values and effect sizes
    for **all questions vs Q85**.
    """
)

# ── 1  Upload widgets ────────────────────────────────────────────────────────
resp_file = st.file_uploader(
    "① Survey responses (.xlsx, raw Qualtrics export)", type="xlsx"
)
map_file = st.file_uploader(
    "② Mapping / question‑type sheet (.xlsx)", type="xlsx"
)

run_btn = st.button("▶️ Run analysis", disabled=not (resp_file and map_file))


# ── Helper: extract Qualtrics code from the column header ────────────────────
def extract_code(label: str) -> str | None:
    """
    "(Q101_2) 2. AI‑generated output …"  ->  "Q101_2"
    Returns None if the pattern isn't found.
    """
    if label.startswith("(") and ")" in label:
        return label[1 : label.find(")")]
    return None


# ── Core statistics engine ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_q85_stats(responses: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    # Build lookup: { "Q101_2": "(Q101_2) ..." }
    code_to_col = {
        extract_code(col): col for col in responses.columns if extract_code(col)
    }

    q85_col = code_to_col.get("Q85")
    if q85_col is None:
        raise ValueError("Column for Q85 not found in the responses file.")

    q85_num = pd.to_numeric(responses[q85_col], errors="coerce")

    out_rows = []
    for _, mrow in mapping.iterrows():
        qcode = mrow["Question"]
        if qcode == "Q85":       # skip self‑comparison
            continue

        col_name = code_to_col.get(qcode)
        if col_name is None:
            continue             # column absent in this export

        series_raw = responses[col_name]
        qtype = str(mrow["Question Type"]).lower()

        # Decide statistical test
        if "nominal" in qtype:
            # ── Kruskal‑Wallis for unordered categories
            groups = [
                q85_num[(series_raw == val) & q85_num.notna()].dropna()
                for val in series_raw.unique()
                if (series_raw == val).sum() >= 2
            ]
            if len(groups) >= 2:
                h, p = stats.kruskal(*groups)
                eff = h                       # we report H as “effect size”
            else:
                p = np.nan
                eff = np.nan
        else:
            # ── Spearman ρ for ordinal, numeric, binaries
            series_num = pd.to_numeric(series_raw, errors="coerce")
            mask = series_num.notna() & q85_num.notna()
            if mask.sum() >= 3:
                eff, p = stats.spearmanr(series_num[mask], q85_num[mask])
            else:
                eff = p = np.nan

        out_rows.append(
            dict(
                Question=qcode,
                Question_Text=mrow["Question Text"],
                Question_Type=mrow["Question Type"],
                Effect_Size=eff,
                p_value=p,
            )
        )

    return pd.DataFrame(out_rows)


# ── 2  Run the analysis when the button is pressed ───────────────────────────
if run_btn:
    try:
        with st.spinner("⏳ Crunching numbers…"):
            df_resp = pd.read_excel(resp_file)
            df_map = pd.read_excel(
                map_file, usecols=["Question", "Question Text", "Question Type"]
            )

            df_out = compute_q85_stats(df_resp, df_map)

        st.success("✅ Done!")
        st.dataframe(
            df_out.style.format({"Effect_Size": "{:.3f}", "p_value": "{:.4f}"}),
            use_container_width=True,
        )

        # Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Q85 statistics")

        st.download_button(
            "💾 Download results as Excel",
            data=buffer.getvalue(),
            file_name="GenAI_Q85_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
