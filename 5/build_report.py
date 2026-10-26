import os, sys, json
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        if self._pageNumber == 1:
            return  # No footer/header on title page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header
        self.drawString(2 * cm, A4[1] - 1.5 * cm, "Fundamentals of Data Science | Case Study 5: Wine Quality Regression")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(2 * cm, A4[1] - 1.6 * cm, A4[0] - 2 * cm, A4[1] - 1.6 * cm)
        # Footer
        self.line(2 * cm, 1.8 * cm, A4[0] - 2 * cm, 1.8 * cm)
        self.drawString(2 * cm, 1.3 * cm, "CBIT Hyderabad | B.E. CSE III Year")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(A4[0] - 2 * cm, 1.3 * cm, page_str)
        self.restoreState()

def build_pdf(student_name="<NAME>", roll_no="<ROLL NO>", section="<SECTION>", faculty="<FACULTY NAME>", github_url="<GITHUB URL>"):
    # Read metrics & results
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    stats_df = pd.read_csv('results/task1_stats.csv')
    cv_df = pd.read_csv('results/task1_cv.csv')
    pearson = metrics['pearson_alcohol_quality']
    rank_df = pd.read_csv('results/task2_feature_correlation_rank.csv')
    reg_df = pd.read_csv('results/task3_regression_metrics.csv')
    std_coef_df = pd.read_csv('results/task3_standardized_coefs.csv')
    vif_df = pd.read_csv('results/task3_vif.csv')

    os.makedirs('report', exist_ok=True)
    clean_roll = roll_no.replace('<', '').replace('>', '').strip()
    pdf_path = f"report/CS5_Wine_Quality_Regression_{clean_roll if clean_roll and clean_roll != 'ROLL NO' else 'Report'}.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2.2*cm
    )

    styles = getSampleStyleSheet()
    normal = styles['Normal']
    normal.fontSize = 9.5
    normal.leading = 13.5
    normal.textColor = colors.HexColor("#222222")

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#722f37"),
        alignment=1
    )

    h1 = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#722f37"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2 = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#8e3a43"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=normal,
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=6
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=normal,
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        alignment=0
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=normal,
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=0
    )

    story = []

    # ================= 1. TITLE PAGE =================
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("Chaitanya Bharathi Institute of Technology (CBIT)", ParagraphStyle('College', fontName='Helvetica-Bold', fontSize=14, leading=18, alignment=1, textColor=colors.HexColor("#333333"))))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Department of Computer Science and Engineering", ParagraphStyle('Dept', fontName='Helvetica', fontSize=11, leading=15, alignment=1, textColor=colors.HexColor("#666666"))))
    story.append(Spacer(1, 1.5 * cm))
    
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#722f37"), spaceAfter=20))
    story.append(Paragraph("CASE STUDY 5: WINE QUALITY PREDICTION", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Statistical Analysis and Multiple Linear Regression on Physicochemical Properties", ParagraphStyle('Sub', fontName='Helvetica-Oblique', fontSize=11, leading=15, alignment=1, textColor=colors.HexColor("#444444"))))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#722f37"), spaceBefore=20, spaceAfter=25))

    story.append(Spacer(1, 1.5 * cm))

    info_data = [
        [Paragraph("<b>Student Name:</b>", table_cell), Paragraph(student_name, table_cell)],
        [Paragraph("<b>Roll Number:</b>", table_cell), Paragraph(roll_no, table_cell)],
        [Paragraph("<b>Section:</b>", table_cell), Paragraph(section, table_cell)],
        [Paragraph("<b>Academic Year:</b>", table_cell), Paragraph("III Year, V Semester (2026-27)", table_cell)],
        [Paragraph("<b>Course / Subject:</b>", table_cell), Paragraph("Fundamentals of Data Science (FDS)", table_cell)],
        [Paragraph("<b>Faculty In-Charge:</b>", table_cell), Paragraph(faculty, table_cell)],
        [Paragraph("<b>Institution:</b>", table_cell), Paragraph("CBIT, Gandipet, Hyderabad - 500075", table_cell)],
        [Paragraph("<b>Date of Submission:</b>", table_cell), Paragraph("26-10-2026", table_cell)]
    ]
    info_table = Table(info_data, colWidths=[4.5*cm, 7.5*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fdf8f8")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2d1d3")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#f0e4e5")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ================= 2. PROBLEM STATEMENT & OBJECTIVES =================
    story.append(Paragraph("1. Problem Statement and Objectives", h1))
    story.append(Paragraph(
        "Wine quality assessment traditionally relies on certified human sensory panels, which is time-intensive, expensive, "
        "and subject to taster fatigue and subjectivity. The objective of this case study is to investigate whether physicochemical laboratory "
        "measurements can predict wine quality scores using statistical tests and multiple linear regression.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Key goals include: (1) conducting descriptive and inferential statistical evaluations for core chemical markers (alcohol, volatile acidity, "
        "residual sugar), (2) evaluating dispersion and testing the significance of linear relationships via hypothesis testing, "
        "(3) exploring the feature space and checking for collinearity, (4) training a multiple linear regression model on an 80:20 train-test split, "
        "and (5) providing evidence-backed business recommendations regarding the feasibility of replacing or augmenting human sensory panels.",
        normal
    ))

    # ================= 3. DATASET DESCRIPTION =================
    story.append(Paragraph("2. Dataset Description", h1))
    story.append(Paragraph(
        "The analysis is conducted on the red wine variant of the Portuguese 'Vinho Verde' dataset compiled by Cortez et al. (2009). "
        "The red wine dataset contains 1,599 observations across 11 continuous physicochemical predictor variables and 1 integer quality target "
        "(median score assigned by at least three sensory experts on a 0 to 10 scale). "
        "The red wine subset was selected as the primary dataset because mixing red and white wines would require an artificial dummy variable "
        "and conflate two distinct chemical profiles.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Data hygiene checks revealed 0 missing values and 240 duplicate rows (15.01%). These duplicate rows were dropped prior to modelling "
        "to prevent data leakage across train/test splits and guard against artificially optimistic evaluation metrics. "
        "The resulting modelling dataset consists of 1,359 clean, unique wine samples.",
        normal
    ))

    # ================= 4. METHODOLOGY =================
    story.append(Paragraph("3. Methodology", h1))
    story.append(Paragraph(
        "The methodological workflow integrates rigorous statistical testing with supervised predictive modelling. "
        "First, descriptive statistics (mean, median, mode, std, skewness) and coefficient of variation (CV) were calculated. "
        "A formal two-tailed Pearson correlation test was executed with alpha = 0.05. "
        "For regression modelling, the dataset was split 80:20 (1,087 train samples, 272 test samples) with random_state=42. "
        "To prevent data leakage, StandardScaler was fitted strictly on training features and subsequently used to transform the test set.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/00_methodology_flowchart.png'):
        img = Image('figures/00_methodology_flowchart.png', width=14*cm, height=7*cm)
        story.append(img)
        story.append(Paragraph("Figure 1: End-to-end wine quality regression and diagnostic workflow", caption_style))

    story.append(Paragraph(
        "Multicollinearity was diagnosed prior to coefficient interpretation using Variance Inflation Factor (VIF) and statsmodels OLS p-values. "
        "Model evaluation was conducted on unseen test data using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), "
        "and R-squared (R2).",
        normal
    ))
    story.append(PageBreak())

    # ================= 5. STATISTICAL FINDINGS (PAGE 3) =================
    story.append(Paragraph("4. Statistical Findings & Exploratory Data Analysis", h1))
    story.append(Paragraph(
        "<b>Task 1a & 1b: Central Tendency, Dispersion, and Relative Variability</b><br/>"
        "Descriptive statistics were computed for alcohol, volatile acidity, and residual sugar (Table 1). "
        "Alcohol exhibited moderate right-skewness (+0.86) with mean 10.43% and median 10.20%. "
        "Volatile acidity showed right-skewness (+0.73) with mean 0.53 g/dm3 and median 0.52. "
        "Residual sugar displayed severe positive skewness (+4.55), reflecting that the majority of Vinho Verde wines are dry "
        "(median 2.20 g/dm3, mode 2.00 g/dm3), with a sparse tail of sweet dessert-style outliers.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Table 1: Task 1a stats
    story.append(Paragraph("<b>Table 1: Descriptive Statistics and Skewness (Task 1a)</b>", h2))
    table_rows = [
        [Paragraph("<b>Attribute</b>", table_header), Paragraph("<b>Mean</b>", table_header),
         Paragraph("<b>Median</b>", table_header), Paragraph("<b>Mode</b>", table_header),
         Paragraph("<b>Std Dev</b>", table_header), Paragraph("<b>Skewness</b>", table_header)]
    ]
    for _, row in stats_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Attribute']), table_cell),
            Paragraph(f"{row['Mean']:.4f}", table_cell),
            Paragraph(f"{row['Median']:.2f}", table_cell),
            Paragraph(f"{row['Mode']:.2f}", table_cell),
            Paragraph(f"{row['Std_Dev']:.4f}", table_cell),
            Paragraph(f"{row['Skewness']:.4f}", table_cell)
        ])
    t1 = Table(table_rows, colWidths=[3.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.5*cm, 2.5*cm])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#722f37")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fdf8f8")]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        f"Coefficient of variation (CV) analysis revealed: Alcohol CV = {cv_df.loc[cv_df['Attribute']=='alcohol', 'CV(%)'].values[0]:.2f}%, "
        f"Volatile Acidity CV = {cv_df.loc[cv_df['Attribute']=='volatile acidity', 'CV(%)'].values[0]:.2f}%, and "
        f"Residual Sugar CV = {cv_df.loc[cv_df['Attribute']=='residual sugar', 'CV(%)'].values[0]:.2f}%. "
        f"Residual sugar exhibits the greatest relative variability. CV is the appropriate comparative measure because standard deviation is scale-dependent, "
        f"whereas CV normalizes dispersion by the mean, permitting valid comparison across differing physical units (% vol vs g/dm3).",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        f"<b>Task 1c: Pearson Correlation Hypothesis Test</b><br/>"
        f"A two-tailed hypothesis test was conducted on alcohol vs quality (H0: r = 0 vs H1: r != 0). "
        f"The Pearson correlation was r = {pearson['r']:.4f} with p-value = {pearson['p_value']:.2e}. "
        f"Since p < 0.05, we reject H0 and conclude that a statistically significant, moderate positive linear relationship exists.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/01_quality_distribution.png'):
        img = Image('figures/01_quality_distribution.png', width=13*cm, height=6.2*cm)
        story.append(img)
        story.append(Paragraph("Figure 2: Wine quality score distribution showing severe concentration around scores 5 and 6 (81.8% of wines)", caption_style))

    story.append(PageBreak())

    # ================= 6. EDA CONTINUED (PAGE 4) =================
    story.append(Paragraph("Task 2b: Alcohol Distribution Grouped by Quality Score", h2))
    story.append(Paragraph(
        "Boxplots illustrate a systematic upward trend in median alcohol content across quality grades: "
        "median alcohol increases steadily from 9.60% at score 5 to 10.50% at score 6, 11.60% at score 7, and 12.50% at score 8. "
        "A series of high-alcohol outliers appears in mid-tier wines (reaching up to 14.9%), indicating that high alcohol is necessary "
        "but not sufficient on its own to achieve superior sensory scores.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/02_alcohol_by_quality_boxplot.png'):
        img = Image('figures/02_alcohol_by_quality_boxplot.png', width=13*cm, height=6.2*cm)
        story.append(img)
        story.append(Paragraph("Figure 3: Alcohol content distribution across quality tiers with median trend and outliers", caption_style))

    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("Task 2c & 2d: Correlation Heatmap and Predictor Ranking", h2))

    if os.path.exists('figures/03_correlation_heatmap.png'):
        img = Image('figures/03_correlation_heatmap.png', width=12*cm, height=7*cm)
        story.append(img)
        story.append(Paragraph("Figure 4: Correlation heatmap of all 11 physicochemical attributes and quality", caption_style))

    story.append(Paragraph(
        "Correlation ranking (Task 2d) identified the top two quality predictors: (1) Alcohol (r = +0.4803, moderate positive), "
        "and (2) Volatile Acidity (r = -0.3952, moderate negative, reflecting vinegar-like acetic acid spoilage). "
        "Noticeable inter-feature correlations include fixed acidity with citric acid (r = +0.67), fixed acidity with density (r = +0.67), "
        "and fixed acidity with pH (r = -0.68), signalling potential collinearity in multiple regression.",
        normal
    ))
    story.append(PageBreak())

    # ================= 7. MODEL EVALUATION (PAGE 5) =================
    story.append(Paragraph("5. Predictive Modelling and Model Diagnostics", h1))
    story.append(Paragraph(
        "A multiple linear regression model was trained on all 11 physicochemical features with an 80:20 train-test split (1,087 train, 272 test). "
        "Test-set performance yielded MAE = 0.5041 quality points, MSE = 0.4310, and RMSE = 0.6565 quality points. "
        "The test R2 of 0.3915 aligns closely with train R2 (0.3520), verifying that the model generalizes stably without overfitting.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Regression table
    story.append(Paragraph("<b>Table 2: Multiple Linear Regression Performance Metrics (Task 3c)</b>", h2))
    table_rows = [
        [Paragraph("<b>Metric</b>", table_header), Paragraph("<b>Value</b>", table_header), Paragraph("<b>Interpretation</b>", table_header)]
    ]
    for _, row in reg_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Metric']), table_cell),
            Paragraph(str(row['Value']), table_cell),
            Paragraph(str(row['Meaning']), table_cell)
        ])
    t2 = Table(table_rows, colWidths=[3.2*cm, 2.5*cm, 10*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#722f37")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#fdf8f8")]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        "<b>Multicollinearity Diagnosis (VIF Analysis):</b><br/>"
        "Variance Inflation Factor (VIF) evaluation on training data identified moderate-to-high multicollinearity: "
        "Fixed Acidity exhibited VIF = 7.95, and Density had VIF = 6.31 (both VIF > 5). "
        "This collinearity arises because fixed acidity directly influences solution density and pH (r = -0.68). "
        "While overall predictions remain unbiased, individual partial regression slope coefficients for these variables must be interpreted cautiously.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/04_standardized_coefficients.png'):
        img = Image('figures/04_standardized_coefficients.png', width=13*cm, height=6.2*cm)
        story.append(img)
        story.append(Paragraph("Figure 5: Standardized regression coefficients ranked by absolute magnitude with direction coding", caption_style))

    story.append(Paragraph(
        "Standardized coefficients rank relative predictor impact: 1. Alcohol (beta = +0.3340), 2. Volatile Acidity (beta = -0.1785), "
        "3. Sulphates (beta = +0.1485), 4. Total Sulfur Dioxide (beta = -0.1236), and 5. pH (beta = -0.1215). "
        "Statsmodels OLS confirmed that volatile acidity, chlorides, total sulfur dioxide, sulphates, and alcohol are statistically significant (p < 0.05).",
        normal
    ))
    story.append(PageBreak())

    # ================= 8. MODEL DIAGNOSTICS & BUSINESS DECISIONS (PAGE 6) =================
    story.append(Paragraph("Model Diagnostics and Residual Analysis (Task 3e)", h2))

    if os.path.exists('figures/05_predicted_vs_actual_residuals.png'):
        img = Image('figures/05_predicted_vs_actual_residuals.png', width=14.5*cm, height=5.5*cm)
        story.append(img)
        story.append(Paragraph("Figure 6: Predicted vs actual quality with jitter (left) and residual plot showing integer tier lines (right)", caption_style))

    story.append(Paragraph(
        "Predicted vs actual plots reveal that linear regression compresses predictions into a narrow band between 5.0 and 6.5. "
        "Because wine quality is a discrete ordinal target (3 to 8), continuous regression shrinks predictions toward the central training mean (5.63), "
        "underpredicting exceptional wines (score 7-8) and overpredicting poor wines (score 3-4). "
        "The residual plot exhibits distinct parallel downward lines (e = k - y_pred) corresponding to each discrete score integer level.",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))

    # ================= BUSINESS DECISIONS =================
    story.append(Paragraph("6. Business Decision-Making (Task 4)", h1))
    story.append(Paragraph(
        "<b>Task 4a: Recommended Laboratory Screening Measurements</b><br/>"
        "Based on the empirical evidence, the winery should implement automated quality screening on two key chemical markers:<br/>"
        "1. <b>Alcohol Content:</b> Strongest positive correlation (r = +0.4803), largest standardized coefficient (+0.3340), and p < 0.0001. "
        "Higher alcohol corresponds to complete grape maturation, sound sugar fermentation, and desirable body.<br/>"
        "2. <b>Volatile Acidity:</b> Second strongest correlation (r = -0.3952), largest negative standardized coefficient (-0.1785), and p < 0.0001. "
        "Elevated acetic acid indicates bacterial spoilage and poor sanitation.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>Task 4b: Model Reliability for Sensory Panel Replacement</b><br/>"
        "The model is NOT reliable enough to replace expert sensory panels. "
        "With an R2 of 0.3915 (explaining ~39% of variance) and an RMSE of 0.6565 quality points on a discrete scale, "
        "the model frequently misjudges wines by an entire rating tier. "
        "It should be deployed strictly as a pre-screening filter to detect flawed batches before submission to tasting panels.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>Task 4c: Real-World Limitations</b><br/>"
        "1. The dataset is restricted to red Vinho Verde wines from northern Portugal; chemical benchmarks do not generalize to other varietals or regions.<br/>"
        "2. Target quality is an ordinal median score from expert tasters, which embodies sensory subjectivity and noise that linear regression cannot fully capture.",
        normal
    ))
    story.append(PageBreak())

    # ================= 9. CONCLUSION & LIMITATIONS (PAGE 7) =================
    story.append(Paragraph("7. Conclusion and Limitations", h1))
    story.append(Paragraph(
        "This case study successfully characterized the relationship between physicochemical attributes and red wine quality. "
        "Multiple linear regression achieved moderate predictive utility (RMSE = 0.66, R2 = 0.39), identifying alcohol and volatile acidity "
        "as the primary drivers of sensory preference. While valuable for rapid automated vat screening, sensory tasting panels remain indispensable.",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Key limitations include: (1) restriction to red Vinho Verde wines, (2) ordinal nature of sensory scores, (3) severe class imbalance "
        "(scores 5 and 6 dominate), (4) inability of linear models to capture non-linear biochemical interactions, and (5) absence of grape varietal, "
        "vintage, or brand metadata.",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))

    # ================= 10. REFERENCES & ATTRIBUTION =================
    story.append(Paragraph("8. References & Attribution", h1))
    story.append(Paragraph(
        "1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. "
        "<i>Decision Support Systems</i>, Elsevier, 47(4):547-553.<br/>"
        "2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.<br/>"
        "3. Seabold, S., and Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. <i>Proceedings of the 9th Python in Science Conference</i>.<br/>"
        f"4. GitHub Repository: {github_url}<br/><br/>"
        f"<b>Attribution:</b> Code structure drafted with AI assistance (Claude); executed, verified and interpreted by {student_name}. "
        "Dataset source: Cortez et al. (2009), UCI Machine Learning Repository.",
        normal
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated: {pdf_path}")

    # Generate preview PNG images using pymupdf
    try:
        import fitz
        doc_pdf = fitz.open(pdf_path)
        preview_dir = os.path.join(os.path.dirname(pdf_path), 'preview')
        os.makedirs(preview_dir, exist_ok=True)
        for i, page in enumerate(doc_pdf):
            pix = page.get_pixmap(dpi=150)
            pix.save(os.path.join(preview_dir, f'page_{i+1}.png'))
        print(f"Exported {len(doc_pdf)} preview pages to {preview_dir}")
    except Exception as e:
        print(f"Preview generation warning: {e}")

    return pdf_path

if __name__ == '__main__':
    s_name = sys.argv[1] if len(sys.argv) > 1 else "<NAME>"
    r_no = sys.argv[2] if len(sys.argv) > 2 else "<ROLL NO>"
    sec = sys.argv[3] if len(sys.argv) > 3 else "<SECTION>"
    fac = sys.argv[4] if len(sys.argv) > 4 else "<FACULTY NAME>"
    gh = sys.argv[5] if len(sys.argv) > 5 else "<GITHUB URL>"
    build_pdf(s_name, r_no, sec, fac, gh)
