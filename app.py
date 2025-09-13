import streamlit as st
import io, csv, math
import numpy as np
import matplotlib.pyplot as plt

st.title("Wavelength Extractor")

# -----------------------
# Mode toggle (TXT stable / CSV beta)
# -----------------------
mode = st.radio(
    "Choose input type",
    ["TXT (stable)", "CSV (beta)"],
    captions=[
        "Two numeric columns per line, separated by spaces/tabs. Decimal comma or dot."
    ],
    index=0,  # default to TXT for now
)

# -----------------------
# Session-state for uploader reset
# -----------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

# -----------------------
# Parsers (both return List[Tuple[float,float]])
# -----------------------
def parse_txt(uploaded_file):
    rows = []
    for line in io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        a = parts[0].replace(",", ".")
        b = parts[1].replace(",", ".")
        try:
            x = float(a); y = float(b)
            rows.append((x, y))
        except ValueError:
            # skip bad lines
            continue
    return rows

def _sniff_delimiter(sample_text: str) -> str:
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "|", "\t"])
        return dialect.delimiter
    except csv.Error:
        for d in [",", ";", "|", "\t"]:
            if d in sample_text:
                return d
        return ","  # fallback

def parse_csv(uploaded_file):
    """
    CSV -> List[(x,y)]
    - Auto-detect delimiter
    - Accepts decimal comma or dot
    - If columns named x,y (any case) exist, use them.
    - Else pick first two numeric-looking columns row by row.
    """
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []

    delim = _sniff_delimiter("\n".join(lines[:50]))
    reader = csv.reader(lines, delimiter=delim)

    header = None
    peek = []
    try:
        first = next(reader)
        peek.append(first)
        # If row has any non-numeric token (not just empty), treat as header
        def _looks_header(row):
            for cell in row:
                c = (cell or "").strip()
                if not c:
                    continue
                # If cannot coerce with dot/comma normalization, it's likely text
                c_norm = c.replace(".", "").replace(",", "")
                if not c_norm.isdigit() and c.lower() not in ("x","y"):
                    return True
            return False
        if _looks_header(first):
            header = [h.strip() for h in first]
        else:
            # no header; keep first data row to parse later
            pass
    except StopIteration:
        return []

    # Build an iterator that includes the peeked row if it was data
    def _iter_rows():
        if header is None and peek:
            yield peek[0]
        for row in reader:
            yield row

    # If header has x/y, map those columns
    xy_idx = None
    if header:
        low = [c.lower() for c in header]
        try:
            xi = low.index("x")
            yi = low.index("y")
            xy_idx = (xi, yi)
        except ValueError:
            pass

    rows_xy = []
    for row in _iter_rows():
        if not row:
            continue
        # Normalize row length to header length if present
        if header:
            row = row + [""] * max(0, len(header) - len(row))
        # If we know x,y positions, use them
        candidates = []
        if xy_idx:
            candidates = [row[xy_idx[0]], row[xy_idx[1]]]
        else:
            # otherwise, pick first two numeric-looking cells in the row
            for cell in row:
                c = (cell or "").strip()
                if not c:
                    continue
                # normalize '1.234,56' -> '1234,56' then comma->dot
                c_norm = c.replace(".", "").replace(",", ".")
                try:
                    float(c_norm)
                    candidates.append(c)
                    if len(candidates) == 2:
                        break
                except ValueError:
                    continue

        if len(candidates) < 2:
            continue

        a = candidates[0].strip().replace(".", "").replace(",", ".") if "," in candidates[0] else candidates[0].strip().replace(",", ".")
        b = candidates[1].strip().replace(".", "").replace(",", ".") if "," in candidates[1] else candidates[1].strip().replace(",", ".")
        try:
            x = float(a); y = float(b)
            rows_xy.append((x, y))
        except ValueError:
            continue

    return rows_xy

# -----------------------
# Uploader + Clear button (aligned in one row)
# -----------------------
accept_types = ["txt"] if mode.startswith("TXT") else ["csv"]

c1, c2 = st.columns([3, 1])
with c1:
    files = st.file_uploader(
        f"Upload one or more .{accept_types[0]} files",
        type=accept_types,
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}_{accept_types[0]}",  # separate key per mode
    )
with c2:
    # tiny spacer to align the button vertically with the uploader
    st.write("")  # one empty line
    st.write("")  # another empty line
    if st.button("Clear all files", use_container_width=True):
        st.session_state.uploaded_files = None
        st.session_state.uploader_key += 1
        st.rerun()

# Persist current selection
if files:
    st.session_state.uploaded_files = files

files = st.session_state.uploaded_files

# -----------------------
# Parse selected files -> list of (x,y) tuples per file
# -----------------------
if files:
    files = sorted(files, key=lambda f: f.name)
    if mode.startswith("TXT"):
        parsed = [parse_txt(f) for f in files]
    else:
        parsed = [parse_csv(f) for f in files]

    if not parsed or not parsed[0]:
        st.error("First file has no readable (x y) pairs."); st.stop()

    # === Your existing logic continues below unchanged ===
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

        # 4) Plot
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xlabel(f"Time (step={step})")
        ax.set_ylabel(f"Value at {target}")
        ax.grid(True)
        st.pyplot(fig)

        # 5) Show table
        st.subheader("Extracted series")
        st.dataframe(
            [{"Experiment_Time": x, f"Value_at_{target}": y} for x, y in zip(x_vals, y_vals)],
            use_container_width=True
        )

        # 6) Download results (choose format)
        from io import StringIO

        download_format = st.radio(
            "Download format",
            ["CSV (; + ,)", "TXT (space-separated)"],
            horizontal=True
        )

        buf = StringIO()
        if download_format.startswith("CSV"):
            writer = csv.writer(buf, delimiter=";")
            writer.writerow(["Experiment_Time", f"Value_at_{target}"])
            for x, y in zip(x_vals, y_vals):
                y_pt = (str(y).replace(".", ",")) if isinstance(y, float) else y
                writer.writerow([x, y_pt])
            st.download_button(
                "Download CSV",
                data=buf.getvalue(),
                file_name="results.csv",
                mime="text/csv",
            )
        else:
            # plain space-separated TXT
            for x, y in zip(x_vals, y_vals):
                y_pt = (str(y).replace(".", ",")) if isinstance(y, float) else y
                buf.write(f"{x} {y_pt}\n")
            st.download_button(
                "Download TXT",
                data=buf.getvalue(),
                file_name="results.txt",
                mime="text/plain",
            )
