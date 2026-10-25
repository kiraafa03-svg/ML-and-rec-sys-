"""Evidence-gated PDF report following the supplied six-page example.

Five body pages plus references; tables contain measured values only. This
module cannot turn software-fixture metrics into a final competition report.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .data import sha256
from .session import inspect_session
from .validation import verify_results

REPOSITORY = "https://github.com/kiraafa03-svg/ML-and-rec-sys-"
REFERENCE = "https://github.com/redlymood/Machine-Learning-and-Recommender-Systems/tree/e7895a18e57d8111456cead185a0216d3eec9d93/exam"
SOURCES = [
    (
        "Kaggle, sweet.tv - TV program recommender, competition data and evaluation description",
        "https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data",
    ),
    (
        "Y. Hu, Y. Koren and C. Volinsky, Collaborative Filtering for Implicit Feedback Datasets, ICDM, 2008, doi:10.1109/ICDM.2008.22",
        "https://yifanhu.net/PUB/cf.pdf",
    ),
    (
        "NetworkX developers, pagerank: personalization, dangling distributions and convergence, official documentation",
        "https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html",
    ),
    (
        "SciPy developers, scipy.sparse, official documentation",
        "https://docs.scipy.org/doc/scipy/reference/sparse.html",
    ),
    (
        "redlymood, Machine-Learning-and-Recommender-Systems/exam, reference implementation, commit e7895a18e57d8111456cead185a0216d3eec9d93",
        REFERENCE,
    ),
    (
        "Kaggle, Official CLI: competition download, submission and score inspection commands",
        "https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md",
    ),
]


def register_fonts() -> None:
    candidates = [
        (os.getenv("REPORT_FONT_REGULAR", ""), os.getenv("REPORT_FONT_BOLD", "")),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ),
        ("C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
        (
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        ),
    ]
    for regular, bold in candidates:
        if regular and bold and Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("ReportSerif", regular))
            pdfmetrics.registerFont(TTFont("ReportSerif-Bold", bold))
            pdfmetrics.registerFontFamily(
                "ReportSerif",
                normal="ReportSerif",
                bold="ReportSerif-Bold",
                italic="ReportSerif",
                boldItalic="ReportSerif-Bold",
            )
            return
    raise FileNotFoundError(
        "No Unicode serif font found. Set REPORT_FONT_REGULAR and REPORT_FONT_BOLD to TTF font paths"
    )


def create_report(args, _test_mode: bool = False) -> dict:
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    summary = json.loads((args.results_dir / "summary.json").read_text())
    if manifest["data_kind"] != "competition" and not _test_mode:
        raise ValueError(
            "Final PDF requires the actual competition dataset, not synthetic fixture results"
        )
    verified = verify_results(args.cache_dir, args.results_dir, args.submission)
    examples = json.loads((args.results_dir / "examples.json").read_text())
    prediction = json.loads((args.results_dir / "final_prediction.json").read_text())
    if not _test_mode:
        evidence = json.loads(args.evidence.read_text())
        if (
            evidence.get("status") != "COMPLETE"
            or evidence.get("source") != "authenticated Kaggle CLI submissions response"
        ):
            raise ValueError("A returned Kaggle private score is required")
        if evidence["submission_sha256"] != sha256(args.submission) or not evidence.get(
            "downloaded_file_matches"
        ):
            raise ValueError("Kaggle result does not match the designated final submission")
        if evidence.get("provenance") != summary["provenance"]:
            raise ValueError("Kaggle evidence belongs to a different experiment")
        screenshot = args.evidence.parent / evidence["screenshot"]
        if sha256(screenshot) != evidence["screenshot_sha256"]:
            raise ValueError("Kaggle screenshot changed")
        session = inspect_session(args.session_file)
        if not args.personal_verification or not args.personal_verification.is_file():
            raise ValueError(
                "Add your own verification statement; the assistant cannot assert you checked a result"
            )
        personal = args.personal_verification.read_text(encoding="utf-8").strip()
        if len(personal) < 60:
            raise ValueError(
                "Personal verification must name a real quantity and method, in your own words"
            )
    else:
        evidence = None
        screenshot = None
        session = None
        personal = "SYNTHETIC LAYOUT TEST. No student verification or Kaggle result is asserted."
    register_fonts()
    body = ParagraphStyle("Body", fontName="ReportSerif", fontSize=10, leading=12, spaceAfter=5)
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="ReportSerif-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=6,
    )
    title_style = ParagraphStyle(
        "Title", parent=heading, fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=12
    )
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10, spaceAfter=3)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.2, leading=10, spaceAfter=0)
    story = []

    def p(text, style=body):
        return Paragraph(text, style)

    def add(text, style=body):
        story.append(p(text, style))

    def table(rows, widths):
        t = Table(
            [[p(str(value), cell) for value in row] for row in rows],
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.black),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 6))

    width = A4[0] - 96
    metrics = verified["independent_map5"]
    selected = summary["selected_config"]["method"]
    gain = metrics["selected"] - metrics["popularity"]
    display = {
        "popularity": "Popularity",
        "ppr": "Personalized PageRank",
        "mf": "Implicit weighted ALS",
        "selected": f"Selected ({selected})",
    }
    labelled = sum(s["labelled_rows"] for s in manifest["schedules"].values())
    scheduled = sum(s["rows"] for s in manifest["schedules"].values())
    coverage = labelled / scheduled if scheduled else 0.0
    split = manifest["splits"]
    cfg = manifest["configuration"]
    add(
        "Leakage-Safe TV Programme Recommendation on sweet.tv:<br/>A Reproducible Comparison of PPR and Implicit Matrix Factorization",
        title_style,
    )
    add(
        f"<b>Student:</b> {escape(args.student)} &nbsp;&nbsp; <b>Team:</b> Individual<br/>"
        f"<b>Email:</b> {escape(args.email or '[not supplied]')} &nbsp;&nbsp; <b>Date:</b> {datetime.now().date().isoformat()}<br/>"
        f"<b>Assignment:</b> Final Exam Quiz - AVITO, LLM4Rec<br/><b>Repository:</b> <link href='{REPOSITORY}' color='blue'>{REPOSITORY}</link>"
    )
    add("Abstract", heading)
    add(
        f"This study predicts five scheduled TV programmes per test user from implicit viewing intervals. "
        "A temporal pipeline compares popularity, programme-graph Personalized PageRank and weighted-ALS matrix factorization; "
        "150 predeclared configurations select an individual method or hybrid on the same holdout. "
        f"The selected strategy achieves validation MAP@5 {metrics['selected']:.5f}, against {metrics['popularity']:.5f} for popularity "
        f"(difference {gain:+.5f}). The result supports a bounded offline recommendation choice; it does not establish user satisfaction or an online business effect."
    )
    add("1. Introduction", heading)
    add(
        "The task is first an interval join, then a forward prediction [1]. Logged session duration is not programme length. "
        "A view must match a labelled airing on the same channel and meet an explicit overlap threshold. "
        "Recurring programmes may legitimately be recommended again; the same repeat policy and cold-start fallback apply to every method. "
        "Contributions are a boundary-safe data pipeline, independently checked computations, a fully recorded search and same-user recommendation comparisons."
    )
    add("2. Related Work", heading)
    add(
        "Implicit-feedback factorization separates observations from confidence [2]. Personalized PageRank provides a restart-conditioned graph ranking and explicit dangling handling [3]. "
        "SciPy sparse operations support the implementation [4]. The supplied exam solution [5] informed the audit and comparison design; its experiment numbers and submissions were not reused. "
        "The present implementations use independent NumPy/SciPy code. Content filtering and association rules were considered, while this bounded implementation focuses on MF and PPR."
    )
    add("3. Method", heading)
    add(
        f"<b>Join:</b> half-open intervals; overlap divided by the airing's own duration; qualifying fraction ≥ {cfg['min_fraction']:.2f}. "
        f"Overlapping-view policy: <b>{cfg['overlap_policy']}</b>. Unlabelled id 0 rows are excluded as items; clipped viewing on their channels may inform schedule context. "
        "Views are clipped before joining, so a late viewing segment cannot enter training through an earlier airing start."
    )
    add(
        "<b>Protocol:</b> history weeks 11-20, holdout 21-30; final refit 11-30, prediction 31-42. Candidates are labelled programmes airing in the target window. "
        "Cold users receive training-popularity Top-5 restricted to that pool; zero-count candidates break ties by ID. All outputs have five unique nonzero IDs."
    )
    add(
        "<b>Models:</b> PPR projects weighted user histories into a programme co-view graph, retains the strongest outgoing edges, and returns dangling mass through the user's restart vector. "
        "ALS minimises confidence-weighted reconstruction error plus factor L2; binary preference and log-count targets are tested. Hybrid weights combine per-user min-max scores or reciprocal ranks, "
        "optionally adjusted by target/history airing exposure on the user's training channels. Scores are ranking signals, not probabilities."
    )
    story.append(PageBreak())
    add("4. Experiments", heading)
    add(
        f"<b>Data accounting:</b> {manifest['views']['rows']:,} input views; {labelled:,}/{scheduled:,} labelled schedule rows ({coverage:.1%}). "
        f"Training has {split['train']['qualifying_airings']:,} qualifying user-airing pairs; validation has {split['validation']['qualifying_airings']:,}. "
        f"The comparison cohort contains {summary['cohort']['evaluable_users']:,} users, including {summary['cohort']['cold_start_users']:,} without qualifying history. "
        f"Excluded target users below the minimum: {summary['cohort']['users_below_minimum_excluded']:,}."
    )
    table(
        [["Method", "Validation MAP@5", "Difference vs popularity"]]
        + [
            [display[m], f"{metrics[m]:.6f}", f"{metrics[m] - metrics['popularity']:+.6f}"]
            for m in metrics
        ],
        [width * 0.48, width * 0.25, width * 0.27],
    )
    add(
        "<b>Metric:</b> AP@5 sums precision at distinct relevant hits and divides by min(5, relevant-set size); MAP is the macro mean. "
        "Targets are each user's five most frequent qualifying programmes. Users need five distinct target programmes for the primary cohort. "
        f"Ties at the fifth target position occur for {summary['cohort']['top5_boundary_ties']:,} users; ascending ID resolves them because the host tie rule is unpublished."
    )
    add(
        "<b>Search:</b> 40 PPR, 70 MF and 40 fusion configurations, fixed in trial_plan.json before execution. "
        f"All 150 completed. Recorded trial time totals {summary['seconds_all_trial_records']:.1f} s; parsing/joining took {manifest['seconds']:.1f} s; "
        f"final prediction took {prediction['seconds']:.1f} s. The holdout is used for selection, so its winning score has selection bias. "
        "Only the returned Kaggle private score is the external competition result; leaderboard position is not used."
    )
    add("Selected configurations", heading)
    for component, value in summary["best_components"].items():
        add(
            f"<b>{component.upper()}:</b> {escape(json.dumps(value['config'], sort_keys=True))}",
            small,
        )
    add(
        f"<b>Final strategy:</b> {escape(json.dumps(summary['selected_config'], sort_keys=True))}",
        small,
    )
    env = manifest["environment"]
    add(
        f"<b>Environment:</b> Python {env['python']}; NumPy {env['numpy']}; SciPy {env['scipy']}; {escape(env['platform'])}. "
        "MF uses seed 42, with one BLAS thread per fit. Full parameters, runtimes, provenance hashes and per-trial status are retained in the repository."
    )
    add("Verification and defect investigation", heading)
    add(
        "Independent checks include AP@5 from hit positions, PPR against a float64 linear-system solution with a dangling node, "
        "ALS objective against a dense oracle and its normal equations, and interval matching against brute force. "
        "Regression fixtures reproduce and prevent the reference's empty-join crash, unseen-user lookup failure, non-monotone-end search failure and cross-boundary training contamination. "
        "These counterexamples establish software defects; their prevalence or score impact in real raw data is not inferred from synthetic fixtures."
    )
    add(
        "A duplicate-view fixture gives summed screen time 1.2 versus unique-time coverage 0.6; the selected overlap policy is disclosed above. "
        "The baseline AP fixture is (1 + 1 + 3/4 + 4/5)/5 = 0.71. Automated artifact checks independently recompute the displayed method metrics and verify template user order and CSV SHA-256."
    )
    for number, (name, user) in enumerate(examples["users"].items(), 1):
        story.append(PageBreak())
        add(f"5.{number} Same-user comparison: {escape(name)} viewer", heading)
        add(
            f"<b>User ID:</b> {escape(user['user_id'])}; <b>cold start:</b> {user['cold_start']}; "
            f"<b>distinct history programmes:</b> {user['history_distinct_programmes']}. "
            "Selection uses history activity percentiles before inspecting recommendation lists; a cold viewer replaces the light example when available."
        )
        history = (
            "; ".join(
                f"{r['id']}: {r['title']} ({r['qualifying_airings']:g})" for r in user["history"]
            )
            or "No qualifying training history."
        )
        add("<b>History:</b> " + escape(history), small)
        actual = "; ".join(f"{r['id']}: {r['title']}" for r in user["actual_top5"])
        add("<b>Actually watched target Top-5:</b> " + escape(actual), small)
        methods = [
            m for m in ["popularity", "ppr", "mf", "selected"] if m in user["recommendations"]
        ]
        rows = [
            [display[m] + f"<br/>AP@5 {user['recommendations'][m]['ap5']:.4f}" for m in methods]
        ]
        for rank in range(5):
            row = []
            for m in methods:
                item = user["recommendations"][m]["top5"][rank]
                row.append(
                    f"<b>{rank + 1}. {item['id']}</b><br/>{escape(item['title'])}<br/>"
                    f"{escape(item['category'])}<br/>hit={int(item['hit'])}; history={int(item['in_history'])}; pop={item['training_popularity']:g}"
                )
            rows.append(row)
        table(rows, [width / len(methods)] * len(methods))
        for method in methods:
            result = user["recommendations"][method]
            repeated = sum(item["in_history"] for item in result["top5"])
            new_hits = sum(item["hit"] and not item["in_history"] for item in result["top5"])
            add(
                f"<b>{display[method]}:</b> {repeated}/5 recommendations appear in this platform's recorded history; "
                f"{new_hits}/5 are held-out hits outside that history. Mean training popularity is {result['mean_training_popularity']:.1f}.",
                small,
            )
        add(
            "Interpretation. MF compresses co-view patterns into shared factors; PPR spreads restart mass over explicit co-view links and retains a direct connection to history. "
            "Their measured hits and familiarity above show whether that difference helps this viewer. A new-to-log item is not automatically useful discovery, "
            "and missing platform history cannot establish that the viewer has never seen a programme elsewhere."
        )
    story.append(PageBreak())
    add("5.3 Final strategy and limitations", heading)
    add(
        f"Select <b>{escape(selected)}</b> under the documented offline objective: its MAP@5 is {metrics['selected']:.5f}, "
        f"a difference of {gain:+.5f} from the common baseline. The individual models remain in the comparison even when their result is weaker. "
        "The two users' ranked IDs, hits and familiarity support the interpretation; production cost is bounded by sparse graph construction, iterative solves and a reproducible refit. "
        "The main limitation is incomplete programme labelling: unlabelled airtime cannot become a recommendation target. "
        "Additional limitations are noisy implicit viewing, cold items, unknown host tie-breaking, temporal drift and repeated use of the holdout for selection. "
        "No live experiment, retention effect or revenue effect is claimed."
    )
    add("Designated Kaggle submission", heading)
    if evidence:
        add(
            f"<b>Submission ref:</b> {evidence['submission_ref']}; <b>private MAP@5:</b> {evidence['private_score']:.5f}; "
            f"<b>public:</b> {evidence['public_score']}. Authenticated CLI output and a byte-identical downloaded submission accompany this record [6]."
        )
        with PILImage.open(screenshot) as image:
            image_width, image_height = image.size
        scale = min(width / image_width, 190 / image_height)
        story.append(
            Image(
                str(screenshot),
                width=image_width * scale,
                height=image_height * scale,
                hAlign="LEFT",
            )
        )
    else:
        add("SYNTHETIC TEST: no Kaggle submission, private score or screenshot exists.")
    add(f"<b>Final CSV SHA-256:</b> {prediction['sha256']}", small)
    add("6. AI Usage Disclosure", heading)
    add(
        "<b>AI assistance:</b> OpenAI Codex in ChatGPT Work was used for source inspection, independent implementation, regression tests, documentation and PDF tooling. "
        "The assistant executed the automated checks. A model build/version is not asserted where the interface did not expose it. "
        "The reference solution [5] is disclosed as external assistance; its code, scores and example-user outcomes are not represented as this project's experiments."
    )
    add("<b>What I personally verified:</b> " + escape(personal))
    add(
        "<b>Trust boundary:</b> automated checks establish only their tested properties. Host-side ground truth, the screenshot's visual content and wider user satisfaction "
        "are not derived from the local test suite. The native session file must be the original work record, not a summary written after execution."
    )
    if session:
        add(
            f"<b>Session export:</b> {escape(session['filename'])}; {session['size_bytes']:,} bytes; "
            f"{session['recognised_tool_calls']} recognised tool calls and {session['recognised_tool_results']} results.<br/>SHA-256: {session['sha256']}",
            small,
        )
    story.append(PageBreak())
    add("References", heading)
    for number, (name, url) in enumerate(SOURCES, 1):
        add(
            f"[{number}] {escape(name)}. [Online]. Available: <link href='{escape(url)}' color='blue'>{escape(url)}</link>. "
            f"[Accessed: {datetime.now().date().isoformat()}]."
        )
    add("Reproduction and artifact map", heading)
    add(
        "exam/data.py: streaming parsing, clipping and exact interval join; models.py: PPR, weighted ALS and fusion; "
        "experiments.py: fixed 150-trial plan and refit; results/trials.csv and trials/*.json: every attempt; "
        "results/examples.json: both users' lists and histories; results/verification.json: independent metric and identity checks; "
        "submissions/final.csv: the exact scored submission. Raw competition data and private session exports are not redistributed through the public repository."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=42,
        bottomMargin=40,
        title="sweet.tv recommendation exam",
        author=args.student,
    )

    def decorate(canvas, document):
        canvas.setFont("ReportSerif", 8)
        canvas.drawCentredString(A4[0] / 2, 22, str(document.page))
        if _test_mode:
            canvas.saveState()
            canvas.setFillColor(colors.red)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.drawCentredString(
                A4[0] / 2, A4[1] - 22, "SYNTHETIC SOFTWARE TEST - NOT AN EXAM REPORT"
            )
            canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return {
        "output": str(args.output),
        "sha256": sha256(args.output),
        "synthetic_layout_test": _test_mode,
    }
