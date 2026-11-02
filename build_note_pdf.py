"""Two-page project note PDF."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Oblique", FD + "DejaVuSans-Oblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique", boldItalic="Sans-Bold")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#616e7c")
ACCENT = colors.HexColor("#1f6feb")
RULE = colors.HexColor("#d9e2ec")
SOFT = colors.HexColor("#f0f4f8")

st = {
    "title": ParagraphStyle("t", fontName="Sans-Bold", fontSize=17, leading=21, textColor=INK),
    "sub": ParagraphStyle("s", fontName="Sans", fontSize=9, leading=12, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Sans-Bold", fontSize=11, leading=14, textColor=ACCENT,
                        spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Sans", fontSize=8.8, leading=12.2, textColor=INK, alignment=TA_LEFT),
    "b": ParagraphStyle("b", fontName="Sans", fontSize=8.8, leading=12.2, textColor=INK, leftIndent=10,
                        bulletIndent=1, spaceAfter=1.5),
    "cell": ParagraphStyle("c", fontName="Sans", fontSize=8, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Sans-Bold", fontSize=8, leading=10.4, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans-Oblique", fontSize=7.6, leading=10, textColor=MUTED),
}
P = lambda t, s="p": Paragraph(t, st[s])
B = lambda t: Paragraph(t, st["b"], bulletText="•")
FIG = "/tmp/claude-0/-home-claude/9a56b2dc-a28f-5a44-bf5e-7f6e733fa0b0/scratchpad/figs/"
W = A4[0] - 30 * mm


def table(rows, widths, header=True):
    data = [[P(c, "cellb" if (header and i == 0) else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    t.setStyle(TableStyle(style))
    return t


def fig(name, width):
    from PIL import Image as PI
    w, h = PI.open(FIG + name).size
    return Image(FIG + name, width=width, height=width * h / w)


def footer(c, doc):
    c.saveState()
    c.setFont("Sans", 7.2)
    c.setFillColor(MUTED)
    c.drawString(15 * mm, 9 * mm, "Shriyam  ·  +91 97980 44462  ·  shriyamgopalgupta@gmail.com  ·  "
                                  "github.com/Shriyam-22/price-forecasting-django-fastapi")
    c.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page} of 2")
    c.restoreState()


s = []
s += [P("Project Note: 12-Month Price Forecasting", "title"), Spacer(1, 2),
      P("AI/ML Developer assignment for Way2Agribusiness India Pvt. Ltd.  ·  Shriyam  ·  "
        "+91 97980 44462  ·  shriyamgopalgupta@gmail.com", "sub"), Spacer(1, 4)]

s += [P("The task", "h"),
      P("Forecast the average monthly price for the next 12 months from 249 months of history "
        "(Jan 2005 – Sep 2025), evaluate the model on both training and future months, and explain how to act "
        "on the forecast and run it in production with Django, FastAPI, Next.js and React Native.")]

s += [P("What I did and how", "h"),
      table([
          ["Step", "How", "Outcome"],
          ["1. Data checks", "Checked the calendar for gaps, nulls, duplicates and invalid prices; searched the series "
           "against every shifted copy of itself for repeated stretches; flagged one-off spikes (more than 22% away from "
           "the 7-month rolling median); listed repeated values.", "No explicit missing values, but hidden problems found (below)."],
          ["2. Cleaning", "Set the bad months to missing and re-estimated them in log space: remove the month-of-year "
           "effect, interpolate the level, add the effect back. Kept real market jumps.",
           "32 months repaired, 9 stale months flagged."],
          ["3. Exploration", "Trend/seasonal decomposition (STL), stationarity tests (ADF, KPSS), autocorrelation of "
           "monthly returns.", "Strong trend, weak seasonality, price close to a random walk."],
          ["4. Modelling", "Compared 10 model families (naive baselines, ETS, Theta, ARIMA/SARIMA, LightGBM) and "
           "grid-searched 71 SARIMA settings. All models work on log prices.", "Selected <b>ARIMA(1,1,0)</b> on log prices."],
          ["5. Evaluation", "13 rolling 12-month backtests (2014–2020) for selection, then a final check on the last "
           "12 months (Oct 2024 – Sep 2025), which no model saw during selection. Metrics: MAPE, RMSE, MASE, bias, "
           "interval coverage.", "8.7% error on the unseen year; all 12 months inside the 95% range."],
          ["6. Forecast", "Refit on all data; forecast Oct 2025 – Sep 2026 with 80% and 95% ranges.",
           "About 15,800, roughly flat, with a wide range."],
          ["7. Deployment", "Built a working Django REST Framework app and a FastAPI service around the model, with "
           "13 automated tests.", "APIs reproduce the notebook forecast exactly."],
      ], [24 * mm, 103 * mm, W - 127 * mm])]

s += [P("Key finding: a hidden copy-paste error in the data", "h"),
      P("Jun 2021 – Nov 2023 (30 months) is an exact, value-for-value copy of Oct 2007 – Aug 2010. It made the price "
        "look like it crashed 53% in one month and then jumped 92% when the copy ended. Treating these months as "
        "missing matters: leaving them in made the price look about 70% more volatile and hid a real momentum effect. "
        "Apr–May 2006 (3,500 then 6,000) was also a data-entry spike and was repaired the same way."),
      Spacer(1, 3), fig("c12.png", W * 0.86),
      P("Orange: the copied months (solid) and the 2007–2010 data they were copied from (dotted).", "cap")]

# ---- page 2
s += [PageBreak(), P("Why ARIMA(1,1,0)?", "h"),
      B("<b>In plain words:</b> next month's % change = about 10% of this month's change + an unpredictable shock. "
        "Monthly shocks average about ±6%, so the forecast quickly settles flat and the uncertainty range widens."),
      B("<b>Most accurate on the unseen year</b> (lowest error of all 10 candidates) and in the top group on the backtests."),
      B("<b>Simple and robust:</b> one parameter, fits in milliseconds, easy to explain and monitor."),
      B("<b>Honest uncertainty:</b> its 95% ranges contained 99% of backtest months and all 12 holdout months."),
      B("<b>LightGBM did worst on the unseen year (16% error):</b> with about 250 data points and only past prices, "
        "it learned false patterns. It would become useful once external drivers are added (mandi arrivals, rainfall, "
        "stocks, policy)."),
      Spacer(1, 4),
      table([["Evaluation", "Average error (MAPE)", "vs seasonal-naive (MASE)", "Inside 95% range"],
             ["Training (1 month ahead)", "4.2%", "–", "–"],
             ["Unseen year, Oct 2024 – Sep 2025", "8.7%", "0.71", "12 of 12 months"],
             ["13 rolling backtests (12 months ahead)", "10.5%", "0.55", "99%"]],
            [W * 0.37, W * 0.2, W * 0.23, W * 0.2])]

s += [P("The forecast: Oct 2025 – Sep 2026", "h"),
      fig("n62.png", W * 0.86),
      P("Central forecast about 15,800, roughly flat. By Sep 2026: 80% chance between about 11,800 and 21,200; "
        "95% chance between about 10,100 and 24,700. Because the price is mostly shocks, the business value is in "
        "planning around the range: hedging part of purchases, staggered buying and selling, warehouse storage, "
        "and alerts when the price leaves the 80% band.")]

s += [P("How it runs in production", "h"),
      B("<b>Django + DRF (system of record):</b> stores monthly prices with data-quality flags, keeps a registry of model "
        "versions, and runs a monthly <i>retrain_forecast</i> job (clean → fit → quality gate → promote). A new model "
        "goes live only if it beats the random-walk benchmark and has honest intervals; old versions stay for rollback."),
      B("<b>FastAPI (real-time predictions):</b> <i>/predict</i> returns the forecast with ranges and accepts newer prices "
        "to update the forecast instantly without retraining. Django calls <i>/reload</i> to swap in a new model with no downtime."),
      B("<b>Next.js and React Native</b> read the same versioned JSON API. Mobile caches the last forecast for poor "
        "connectivity and gets push alerts when the price crosses the forecast range."),
      B("<b>Monitoring:</b> monthly accuracy, bias, interval coverage and volatility drift, with alerts that trigger retraining."),
      Spacer(1, 5),
      P("<b>Files:</b> full notebook <i>Price_Forecasting_Shriyam.ipynb</i>. Code and tests: "
        "github.com/Shriyam-22/price-forecasting-django-fastapi")]

doc = SimpleDocTemplate("/mnt/user-data/outputs/Project_Note_Price_Forecasting_Shriyam.pdf", pagesize=A4,
                        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm, bottomMargin=15 * mm,
                        title="Project Note: 12-Month Price Forecasting", author="Shriyam")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("ok")
