# app.py
import streamlit as st
import io, csv, math
import numpy as np
import matplotlib.pyplot as plt

st.title("Wavelength Extractor")

# 1) Upload multiple .txt files (with Clear button)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

c1, c2 = st.columns([4, 1])
with c1:
    files = st.file_uploader(
        "Upload one or more .txt files",
        type=["txt"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",  # dynamic key
    )
with c2:
    if st.button("Clear all files"):
        st.session_state.uploaded_files = None
        st.session_state.uploader_key += 1   # forces a fresh uploader
        st.rerun()                           # restart script to reflect change

# persist current selection
if files:
    st.session_state.uploaded_files = files

files = st.session_state.uploaded_files

def parse_txt(uploaded_file):
    rows = []
    for line in io.StringIO(uploaded_file.getvalue().decode("utf-8")):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                x = float(parts[0].replace(",", "."))
                y = float(parts[1].replace(",", "."))
                rows.append((x, y))
            except ValueError:
                pass
    return rows

if files:
    files = sorted(files, key=lambda f: f.name)
    parsed = [parse_txt(f) for f in files]
    if not parsed[0]:
        st.error("First file has no readable (x y) pairs."); st.stop()

    # 2) Choose wavelength from first file
    first_wls = [f"{x}".replace(".", ",") for (x, _) in parsed[0]]
    target = st.selectbox("Choose wavelength (from first file)", first_wls)

    # 3) Time step
    step = st.number_input("Time step between experiments", min_value=0.0, value=5.0)

    if target and step > 0:
        tgt = float(target.replace(",", "."))
        x_vals, y_vals = [], []

        for i, rows in enumerate(parsed):
            if not rows:
                x_vals.append(i * step); y_vals.append(math.nan); continue
            xs = np.array([r[0] for r in rows])
            ys = np.array([r[1] for r in rows])
            idx = int(np.argmin(np.abs(xs - tgt)))
            x_vals.append(i * step)
            y_vals.append(float(ys[idx]))

        # 4) Show table
        st.subheader("Extracted series")
        st.dataframe(
            [{"Experiment_Time": x, f"Value_at_{target}": y} for x, y in zip(x_vals, y_vals)],
            use_container_width=True
        )

        # 5) Plot
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xlabel(f"Time (step={step})")
        ax.set_ylabel(f"Value at {target}")
        ax.grid(True)
        st.pyplot(fig)

        # 6) Download CSV
        from io import StringIO
        buf = StringIO()
        w = csv.writer(buf, delimiter=";")
        w.writerow(["Experiment_Time", f"Value_at_{target}"])
        for x, y in zip(x_vals, y_vals):
            y_pt = (str(y).replace(".", ",")) if isinstance(y, float) else y
            w.writerow([x, y_pt])
        st.download_button("Download CSV (; + ,)", data=buf.getvalue(),
                           file_name="results.csv", mime="text/csv")
