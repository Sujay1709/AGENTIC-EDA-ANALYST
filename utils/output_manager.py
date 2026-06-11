import os
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class ReportPDF(FPDF):
    """
    Custom PDF class with header and footer.
    Inherits from FPDF so we can override the built-in header/footer methods.
    """

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(50, 50, 50)
        self.cell(
            0, 10,
            "Agentic EDA Report",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C"
        )
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Page {self.page_no()}",
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C"
        )


def _latin1(text: str) -> str:
    """
    FPDF core fonts only support latin-1. LLM output can contain unicode
    (smart quotes, bullets, >=) that would crash pdf.output(); replace anything
    outside latin-1 with a close ASCII equivalent or '?'.
    """
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "-", "…": "...",
        "≥": ">=", "≤": "<=", "×": "x",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def add_section_title(pdf: ReportPDF, title: str):
    """Adds a bold section heading with a horizontal rule."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(3)


def add_body_text(pdf: ReportPDF, text: str):
    """
    Adds multi-line body text.
    multi_cell handles line wrapping automatically.
    """
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, _latin1(text))
    pdf.ln(2)


def add_plot(pdf: ReportPDF, image_path: str, caption: str = ""):
    """
    Embeds a plot image into the PDF.
    Checks if there's enough space on the current page first.
    """
    if not os.path.exists(image_path):
        print(f"[OutputManager] Warning: Plot not found at {image_path}, skipping.")
        return

    # If less than 100 units remain on page, start a new page
    if pdf.get_y() > 180:
        pdf.add_page()

    pdf.image(image_path, x=10, w=190)

    if caption:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(
            0, 6, caption,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C"
        )
    pdf.ln(4)


# Plot filename prefix -> (sort order, caption label). Keeps related figures
# grouped together in the report and gives each a human-readable caption. These
# prefixes are produced by fallback_plots.generate_fallback_plots.
_PLOT_ORDER = [
    ("dist_", 0, "Distribution"),
    ("box_", 1, "Boxplot"),
    ("correlation_heatmap", 2, "Correlation Heatmap"),
    ("scatter_", 3, "Relationship"),
    ("trend_over_", 4, "Trend over"),
    ("value_count_", 5, "Value Counts"),
]


def _plot_caption(plot_path: str) -> str:
    """Build a friendly caption from a plot filename's prefix."""
    name = os.path.splitext(os.path.basename(plot_path))[0]
    if name == "correlation_heatmap":
        return "Correlation Heatmap"
    if name.startswith("scatter_"):
        body = name[len("scatter_"):].replace("_vs_", " vs ").replace("_", " ")
        return f"Relationship: {body}"
    for prefix, _, label in _PLOT_ORDER:
        if prefix.endswith("_") and name.startswith(prefix):
            return f"{label}: {name[len(prefix):].replace('_', ' ')}"
    return name.replace("_", " ").title()


def _plot_sort_key(plot_path: str) -> int:
    """Order plots by group so related figures sit together in the report."""
    name = os.path.basename(plot_path)
    for prefix, order, _ in _PLOT_ORDER:
        if name.startswith(prefix):
            return order
    return len(_PLOT_ORDER)


def build_report(
    schema: dict,
    insights: str,
    saved_plots: list,
    output_dir: str,
    generated_code: str = "",
    missing_report: dict = None,
    missing_suggestions: str = ""
) -> str:
    """
    Assembles the full EDA PDF report.

    Args:
        schema: Output from schema_extractor
        insights: Text output from analyst_agent
        saved_plots: List of plot file paths from code_executor
        output_dir: Where to save the PDF
        generated_code: The code that was executed (optional, for appendix)
        missing_report: Structured output from missing_analyzer.analyze_missing
        missing_suggestions: Per-column handling recommendations from the missing-values agent

    Returns:
        Path to the saved PDF file
    """
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Title block ---
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(
        0, 12, "Exploratory Data Analysis Report",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    pdf.cell(
        0, 8, f"Generated on {timestamp}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"
    )
    pdf.ln(6)

    # --- Section 1: Dataset Overview ---
    add_section_title(pdf, "1. Dataset Overview")

    overview_lines = [
        f"File: {schema['filepath']}",
        f"Shape: {schema['shape'][0]} rows x {schema['shape'][1]} columns",
        f"Numeric columns: {', '.join(schema['numeric_columns']) or 'None'}",
        f"Categorical columns: {', '.join(schema['categorical_columns']) or 'None'}",
    ]
    add_body_text(pdf, "\n".join(overview_lines))

    # --- Section 2: Missing Values Analysis ---
    add_section_title(pdf, "2. Missing Values Analysis")

    if missing_report:
        summary = (
            f"Total missing cells: {missing_report['total_missing']} "
            f"({missing_report['total_missing_percent']}% of {missing_report['total_cells']} cells)\n"
            f"Columns fully complete: {len(missing_report['complete_columns'])} of "
            f"{len(missing_report['per_column'])}"
        )
        add_body_text(pdf, summary)

        affected = missing_report["affected_columns"]
        if affected:
            breakdown = ["Per-column breakdown (columns with missing data only):"]
            for col, info in affected.items():
                breakdown.append(
                    f"  - {col} [{info['dtype']}]: {info['count']} missing ({info['percent']}%)"
                )
            add_body_text(pdf, "\n".join(breakdown))

            if missing_report["high_missing_columns"]:
                add_body_text(
                    pdf,
                    "Columns >50% missing (consider dropping): "
                    + ", ".join(missing_report["high_missing_columns"]),
                )

            if missing_report["co_missing_pairs"]:
                pair_lines = ["Columns frequently missing together:"]
                for p in missing_report["co_missing_pairs"]:
                    pair_lines.append(
                        f"  - {p['columns'][0]} & {p['columns'][1]} (corr {p['correlation']})"
                    )
                add_body_text(pdf, "\n".join(pair_lines))
        else:
            add_body_text(pdf, "No missing values detected.")
    else:
        # Backward-compatible fallback: list straight from the schema.
        missing_lines = [
            f"  - {col}: {pct}% missing"
            for col, pct in schema["missing_percent"].items() if pct > 0
        ]
        add_body_text(pdf, "\n".join(missing_lines) if missing_lines else "No missing values detected.")

    # Missing-values heatmap (saved as missing_values.png by missing_analyzer)
    heatmap_path = os.path.join(output_dir, "missing_values.png")
    add_plot(pdf, heatmap_path, caption="Missing Values Map")

    # Per-column missing-% bar chart (saved as missing_bar.png by the plotter)
    add_plot(
        pdf,
        os.path.join(output_dir, "missing_bar.png"),
        caption="Missing Values by Column",
    )

    # Per-column handling recommendations from the missing-values agent
    if missing_suggestions:
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, "Recommended Handling (with pandas code)")
        pdf.ln(1)
        add_body_text(pdf, _latin1(missing_suggestions))

    # --- Section 3: Visualizations ---
    # Skip the missing-values figures here — they are already shown in Section 2.
    missing_figs = {"missing_values.png", "missing_bar.png"}
    viz_plots = [p for p in saved_plots if os.path.basename(p) not in missing_figs]
    viz_plots.sort(key=_plot_sort_key)  # group related figures together
    if viz_plots:
        add_section_title(pdf, "3. Visualizations")
        for plot_path in viz_plots:
            add_plot(pdf, plot_path, caption=_plot_caption(plot_path))

    # --- Section 4: AI Insights ---
    add_section_title(pdf, "4. AI-Generated Insights")
    add_body_text(pdf, insights)

    # --- Section 5: Generated Code (Appendix) ---
    if generated_code:
        pdf.add_page()
        add_section_title(pdf, "5. Appendix: Generated Analysis Code")
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, _latin1(generated_code))

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    filename = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(output_dir, filename)
    pdf.output(output_path)

    print(f"[OutputManager] PDF report saved to: {output_path}")
    return output_path