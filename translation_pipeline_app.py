import streamlit as st
import pandas as pd
import io
import zipfile
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="Translation Pipeline Processor", layout="wide", page_icon="🌐")

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .err-badge {
        display:inline-block; background:#fee2e2; color:#991b1b;
        border-radius:20px; padding:3px 12px; font-size:0.8rem; margin:2px;
    }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS = {
    "Openai_score", "Openai_riskLevel", "Openai_clinicalIssues", "Openai_suggestedTranslation",
    "Gemini_score", "Gemini_riskLevel", "Gemini_clinicalIssues", "Gemini_suggestedTranslation",
    "Source (JA)"
}

# ── Pipeline core ─────────────────────────────────────────────────────────────
def jp_char_count(text):
    if pd.isna(text): return 0
    return sum(1 for ch in str(text)
               if '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u309f'
               or '\u30a0' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u4dbf')


def run_pipeline(df: pd.DataFrame, openai_min: int, gemini_min: int, filename: str):
    stats = {"filename": filename, "input_rows": len(df)}

    # Drop summary/empty rows
    df = df[df["Source (JA)"].notna()].copy()

    # Force numeric on score columns — handles strings like "N/A", "0", "High", etc.
    df["Openai_score"] = pd.to_numeric(df["Openai_score"], errors="coerce")
    df["Gemini_score"] = pd.to_numeric(df["Gemini_score"], errors="coerce")

    # Step 1: Drop rows where BOTH OpenAI AND Gemini are 0/blank
    oi = df["Openai_score"].isna() | (df["Openai_score"] < openai_min)
    gi = df["Gemini_score"].isna() | (df["Gemini_score"] == 0)
    df = df[~(oi & gi)].copy()
    stats["after_step1"] = len(df)

    # Step 2: Merge OpenAI → Gemini where Gemini score == 0
    mask0 = df["Gemini_score"] == 0
    for col_g, col_o in [
        ("Gemini_score",               "Openai_score"),
        ("Gemini_riskLevel",            "Openai_riskLevel"),
        ("Gemini_clinicalIssues",       "Openai_clinicalIssues"),
        ("Gemini_suggestedTranslation", "Openai_suggestedTranslation"),
    ]:
        df.loc[mask0, col_g] = df.loc[mask0, col_o]
    stats["gemini_updated"] = int(mask0.sum())

    # Step 3: Drop OpenAI columns
    df = df.drop(columns=["Openai_score", "Openai_riskLevel",
                           "Openai_clinicalIssues", "Openai_suggestedTranslation"])

    # Step 4: Filter Gemini score >= gemini_min
    df = df[df["Gemini_score"] >= gemini_min].copy()
    stats["final_segments"] = len(df)

    # Step 5: JP character count
    df.insert(df.columns.get_loc("Source (JA)") + 1, "JP_CharCount",
              df["Source (JA)"].apply(jp_char_count))

    # Step 6: Tag source filename
    df.insert(0, "Source_File", filename)

    stats["total_jp_chars"] = int(df["JP_CharCount"].sum())
    stats["risk_counts"] = df["Gemini_riskLevel"].value_counts().to_dict()
    return df, stats


def style_excel(df: pd.DataFrame, sheet_name: str = "Linguist_Segments") -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name=sheet_name[:31])
    buf.seek(0)
    wb = load_workbook(buf)
    ws = wb.active

    hdr_fill = PatternFill("solid", fgColor="1a3a6b")
    hdr_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    for cell in ws[1]:
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    risk_fills = {
        "high risk":   PatternFill("solid", fgColor="FFE4E4"),
        "medium risk": PatternFill("solid", fgColor="FFF8E1"),
        "low risk":    PatternFill("solid", fgColor="E8F5E9"),
    }
    risk_col = next((i for i, c in enumerate(ws[1], 1)
                     if c.value and "riskLevel" in str(c.value).lower()), None)

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for row in ws.iter_rows(min_row=2):
        risk_val = str(row[risk_col - 1].value).lower() if risk_col else ""
        fill = next((v for k, v in risk_fills.items() if k in risk_val), None)
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = border
            if fill:
                cell.fill = fill

    for col in ws.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 45)

    ws.freeze_panes = "A2"
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def build_zip(results: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        frames = []
        for df, stats in results:
            fname = stats["filename"].replace(".xlsx", "")
            zf.writestr(f"{fname}_processed.xlsx",
                        style_excel(df, sheet_name=fname[:31]))
            frames.append(df)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            zf.writestr("ALL_FILES_MERGED.xlsx",
                        style_excel(merged, "All_Segments"))
            zf.writestr("ALL_FILES_MERGED.csv",
                        merged.to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    openai_min = st.slider("Min OpenAI score", 1, 10, 1)
    gemini_min = st.slider("Min Gemini score (→ linguists)", 1, 10, 3)
    st.caption("Rows where both scores are below threshold are dropped.")
    st.divider()
    st.markdown("""
**Pipeline Steps**
1. 🔍 Drop rows where both scores ≤ threshold
2. 🔄 Merge OpenAI → Gemini (where Gemini = 0)
3. 🗑️ Delete OpenAI columns
4. ✂️ Keep Gemini score ≥ min
5. 🈳 Add JP character count
6. 🏷️ Tag source filename
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌐 Translation Pipeline Processor</div>',
            unsafe_allow_html=True)
st.markdown("Upload **one or many** pipeline Excel files — processed individually and merged.")
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Pipeline Files")
uploaded_files = st.file_uploader(
    "Drop one or multiple .xlsx files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="All files must have the same column structure as the pipeline template."
)

if not uploaded_files:
    st.info("👆 Upload one or more Excel pipeline files to begin.")
    with st.expander("ℹ️ What does this app do?"):
        st.markdown("""
**Supports bulk upload** — drop as many `.xlsx` pipeline files as needed.

Each file goes through the same pipeline:
1. Filter out rows where both OpenAI & Gemini scores are 0/blank
2. Merge OpenAI data into Gemini columns where Gemini score = 0
3. Remove all OpenAI columns
4. Keep only Gemini score ≥ threshold (default 3)
5. Add Japanese character count per segment
6. Tag each row with its source filename

**Outputs:**
- Per-file download (Excel + CSV)
- Combined merged Excel + CSV
- All files zipped into one download
        """)
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
results, errors = [], []
progress = st.progress(0, text="Processing files…")

for i, f in enumerate(uploaded_files):
    try:
        df_raw = pd.read_excel(f, header=0)
        missing = REQUIRED_COLS - set(df_raw.columns)
        if missing:
            errors.append((f.name, f"Missing columns: {', '.join(sorted(missing))}"))
        else:
            df_out, stats = run_pipeline(df_raw, openai_min, gemini_min, f.name)
            results.append((df_out, stats))
    except Exception as e:
        errors.append((f.name, str(e)))
    progress.progress((i + 1) / len(uploaded_files), text=f"Processed: {f.name}")

progress.empty()

if errors:
    st.error(f"⚠️ {len(errors)} file(s) skipped due to errors:")
    for fname, msg in errors:
        st.markdown(f'<span class="err-badge">❌ {fname}</span> — {msg}',
                    unsafe_allow_html=True)

if not results:
    st.warning("No files were processed successfully. Please check your uploads.")
    st.stop()

# ── Global metrics ────────────────────────────────────────────────────────────
st.markdown("### 📊 Overall Summary")
total_input    = sum(s["input_rows"]     for _, s in results)
total_final    = sum(s["final_segments"] for _, s in results)
total_updated  = sum(s["gemini_updated"] for _, s in results)
total_jp_chars = sum(s["total_jp_chars"] for _, s in results)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📁 Files Processed",      len(results))
c2.metric("📥 Total Input Rows",     total_input)
c3.metric("🔄 Gemini Cells Updated", total_updated)
c4.metric("✂️ Final Segments",       total_final)
c5.metric("🈳 Total JP Chars",       f"{total_jp_chars:,}")

# ── Risk chart (all files) ────────────────────────────────────────────────────
all_dfs  = pd.concat([df for df, _ in results], ignore_index=True)
all_dfs["Gemini_score"] = pd.to_numeric(all_dfs["Gemini_score"], errors="coerce").fillna(0)
risk_agg = all_dfs["Gemini_riskLevel"].value_counts().reset_index()
risk_agg.columns = ["Risk Level", "Count"]

rc1, rc2 = st.columns([1, 2])
with rc1:
    st.markdown("**Risk Breakdown — all files**")
    st.dataframe(risk_agg, use_container_width=True, hide_index=True)
with rc2:
    st.bar_chart(risk_agg.set_index("Risk Level"))

st.divider()

# ── Per-file tabs + merged tab ────────────────────────────────────────────────
st.markdown("### 📑 Results by File")
tab_labels = [s["filename"][:28] for _, s in results]
tabs = st.tabs(["🗂️ ALL MERGED"] + tab_labels)

# ALL MERGED tab
with tabs[0]:
    st.markdown(f"**{len(all_dfs)} segments across {len(results)} file(s)**")
    fa, fb, fc = st.columns(3)
    with fa:
        smin_v = int(all_dfs["Gemini_score"].min())
        smax_v = int(all_dfs["Gemini_score"].max())
        if smin_v < smax_v:
            sr_m = st.slider("Score range", smin_v, smax_v, (smin_v, smax_v), key="m_sr")
        else:
            st.caption(f"Score range: all segments scored **{smin_v}**")
            sr_m = (smin_v, smax_v)
    with fb:
        ro_m = ["All"] + sorted(all_dfs["Gemini_riskLevel"].dropna().unique().tolist())
        rs_m = st.selectbox("Risk Level", ro_m, key="m_rl")
    with fc:
        fo_m = ["All"] + [s["filename"] for _, s in results]
        fs_m = st.selectbox("Source File", fo_m, key="m_fi")

    dv = all_dfs.copy()
    dv = dv[(dv["Gemini_score"] >= sr_m[0]) & (dv["Gemini_score"] <= sr_m[1])]
    if rs_m != "All": dv = dv[dv["Gemini_riskLevel"] == rs_m]
    if fs_m != "All": dv = dv[dv["Source_File"] == fs_m]

    st.dataframe(dv, use_container_width=True, height=420)
    st.caption(f"Showing **{len(dv)}** of **{len(all_dfs)}** segments")

# Individual file tabs
for idx, (tab, (df, stats)) in enumerate(zip(tabs[1:], results)):
    with tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input Rows",     stats["input_rows"])
        m2.metric("Gemini Updated", stats["gemini_updated"])
        m3.metric("Final Segments", stats["final_segments"])
        m4.metric("JP Chars",       f"{stats['total_jp_chars']:,}")

        rdf = pd.DataFrame(stats["risk_counts"].items(), columns=["Risk Level", "Count"])
        r1, r2 = st.columns([1, 2])
        with r1: st.dataframe(rdf, hide_index=True, use_container_width=True)
        with r2:
            if not rdf.empty: st.bar_chart(rdf.set_index("Risk Level"))

        f1, f2, f3 = st.columns(3)
        with f1:
            sv = int(df["Gemini_score"].min()) if len(df) else 0
            xv = int(df["Gemini_score"].max()) if len(df) else 10
            if sv < xv:
                sr = st.slider("Score range", sv, xv, (sv, xv), key=f"sr_{idx}")
            else:
                st.caption(f"Score range: all segments scored **{sv}**")
                sr = (sv, xv)
        with f2:
            ro = ["All"] + sorted(df["Gemini_riskLevel"].dropna().unique().tolist())
            rs = st.selectbox("Risk Level", ro, key=f"rl_{idx}")
        with f3:
            st_txt = st.text_input("🔍 Search Source (JA)", key=f"s_{idx}")

        dp = df[(df["Gemini_score"] >= sr[0]) & (df["Gemini_score"] <= sr[1])]
        if rs != "All":    dp = dp[dp["Gemini_riskLevel"] == rs]
        if st_txt:         dp = dp[dp["Source (JA)"].astype(str).str.contains(st_txt, na=False)]

        st.dataframe(dp, use_container_width=True, height=380)
        st.caption(f"Showing **{len(dp)}** of **{stats['final_segments']}** segments")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Excel", key=f"dx_{idx}",
                data=style_excel(df, sheet_name=stats["filename"][:31].replace(".xlsx", "")),
                file_name=stats["filename"].replace(".xlsx", "_processed.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                "⬇️ CSV", key=f"dc_{idx}",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=stats["filename"].replace(".xlsx", "_processed.csv"),
                mime="text/csv",
                use_container_width=True
            )

st.divider()

# ── Bulk export ───────────────────────────────────────────────────────────────
st.markdown("### 📦 Bulk Export")
e1, e2, e3 = st.columns(3)

with e1:
    st.download_button(
        "⬇️ Download All as ZIP",
        data=build_zip(results),
        file_name="pipeline_processed_all.zip",
        mime="application/zip",
        use_container_width=True
    )
with e2:
    st.download_button(
        "⬇️ Merged Excel",
        data=style_excel(all_dfs, "All_Segments"),
        file_name="ALL_FILES_MERGED.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
with e3:
    st.download_button(
        "⬇️ Merged CSV",
        data=all_dfs.to_csv(index=False).encode("utf-8-sig"),
        file_name="ALL_FILES_MERGED.csv",
        mime="text/csv",
        use_container_width=True
    )

st.success(
    f"✅ **{len(results)} file(s) processed** · "
    f"**{total_final} segments** ready for linguists · "
    f"**{total_jp_chars:,} JP characters** total"
)