"""
CBIT Format Comprehensive Case Study Report Generator
Builds a professional, academic PDF report matching the exact format of:
FDS_Case_Studies_3_and_6_160124733191_CBIT_format.pdf
"""
import os
import sys
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas
import fitz  # PyMuPDF for page inspection and preview rendering

class CBITCanvas(canvas.Canvas):
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
            self.draw_page_decorations()
            super().showPage()
        super().save()

    def draw_page_decorations(self):
        self.saveState()
        self.setFont("Times-Roman", 10)
        self.setFillColor(colors.black)

        page_num = self._pageNumber
        width, height = A4

        # Page 1: Title page (no number)
        if page_num == 1:
            pass
        # Pages 2, 3, 4: Front matter with Roman numerals (ii, iii, iv)
        elif page_num == 2:
            self.drawCentredString(width / 2.0, 1.5 * cm, "ii")
        elif page_num == 3:
            self.drawCentredString(width / 2.0, 1.5 * cm, "iii")
        elif page_num == 4:
            self.drawCentredString(width / 2.0, 1.5 * cm, "iv")
        # Pages 5+: Main content with Arabic numerals starting at 1
        else:
            arabic_page = page_num - 4
            self.drawCentredString(width / 2.0, 1.5 * cm, str(arabic_page))

        self.restoreState()

def create_report_story(student_name="Katthi Goutham Reddy", roll_no="160124733179", section="CSE-3", 
                        faculty="Dr. S. Durga Devi", github_url="https://github.com/katthi-goutham-reddy/FDS-case-study",
                        toc_pages=None, lof_pages=None):
    
    # Load Results DataFrames
    cs2_stats_df = pd.read_csv('results/cs2_summary_stats.csv')
    cs2_comp_df = pd.read_csv('results/cs2_clustering_comparison.csv')
    cs2_prof_df = pd.read_csv('results/cs2_cluster_profiles.csv')
    
    cs5_stats_full_df = pd.read_csv('results/cs5_summary_stats.csv')
    cs5_stats_df = pd.read_csv('results/cs5_task1_stats.csv')
    cs5_cv_df = pd.read_csv('results/cs5_task1_cv.csv')
    cs5_rank_df = pd.read_csv('results/cs5_task2_feature_correlation_rank.csv')
    cs5_reg_df = pd.read_csv('results/cs5_task3_regression_metrics.csv')
    cs5_vif_df = pd.read_csv('results/cs5_task3_vif.csv')
    cs5_ols_df = pd.read_csv('results/cs5_task3_ols_summary.csv') if os.path.exists('results/cs5_task3_ols_summary.csv') else None
    cs5_cv_stab_df = pd.read_csv('results/cs5_task3_cv_stability.csv') if os.path.exists('results/cs5_task3_cv_stability.csv') else None

    # Typography Styles matching CBIT reference
    styles = getSampleStyleSheet()

    body = ParagraphStyle(
        'CBITBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=14.5,
        textColor=colors.black,
        alignment=4,  # Justified
        spaceBefore=2.5,
        spaceAfter=2.5
    )

    body_left = ParagraphStyle(
        'CBITBodyLeft',
        parent=body,
        alignment=0  # Left aligned
    )

    bullet_style = ParagraphStyle(
        'CBITBullet',
        parent=body,
        fontSize=10.0,
        leading=14.0,
        leftIndent=12,
        firstLineIndent=-8,
        spaceBefore=2,
        spaceAfter=2
    )

    ch_title = ParagraphStyle(
        'CBITChTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.black,
        alignment=0,
        spaceBefore=4,
        spaceAfter=7,
        keepWithNext=True
    )

    sec_title = ParagraphStyle(
        'CBITSecTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.black,
        alignment=0,
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True
    )

    subsec_title = ParagraphStyle(
        'CBITSubSecTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.black,
        alignment=0,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    fig_caption = ParagraphStyle(
        'CBITFigCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceBefore=3,
        spaceAfter=6
    )

    table_caption = ParagraphStyle(
        'CBITTableCaption',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True
    )

    tbl_header = ParagraphStyle(
        'CBITTblHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        alignment=1
    )

    tbl_cell = ParagraphStyle(
        'CBITTblCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        alignment=1
    )

    tbl_cell_left = ParagraphStyle(
        'CBITTblCellLeft',
        parent=tbl_cell,
        alignment=0
    )

    story = []

    # ================= PAGE 1: TITLE PAGE (i) =================
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("A Case Study Report", ParagraphStyle('TP1', fontName='Times-Bold', fontSize=13, leading=17, alignment=1)))
    story.append(Paragraph("on", ParagraphStyle('TP2', fontName='Times-Roman', fontSize=11, leading=15, alignment=1)))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("“Case Study 2: Customer Segmentation using Clustering Analysis on East-West Airlines”", 
                           ParagraphStyle('TP3', fontName='Times-Bold', fontSize=13, leading=18, alignment=1)))
    story.append(Paragraph("and", ParagraphStyle('TP4', fontName='Times-Roman', fontSize=11, leading=15, alignment=1)))
    story.append(Paragraph("“Case Study 5: Wine Quality Prediction via Multiple Linear Regression”", 
                           ParagraphStyle('TP5', fontName='Times-Bold', fontSize=13, leading=18, alignment=1)))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("for the course", ParagraphStyle('TP6', fontName='Times-Roman', fontSize=10.5, leading=14, alignment=1)))
    story.append(Paragraph("Fundamentals of Data Science (22CSC59)", ParagraphStyle('TP7', fontName='Times-Bold', fontSize=14, leading=18, alignment=1)))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Submitted in partial fulfillment of the requirements for the award of degree", 
                           ParagraphStyle('TP8', fontName='Times-Roman', fontSize=9.5, leading=13, alignment=1)))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("BACHELOR OF ENGINEERING", ParagraphStyle('TP9', fontName='Times-Bold', fontSize=13, leading=17, alignment=1)))
    story.append(Paragraph("in", ParagraphStyle('TP10', fontName='Times-Roman', fontSize=10.5, leading=14, alignment=1)))
    story.append(Paragraph("COMPUTER SCIENCE AND ENGINEERING", ParagraphStyle('TP11', fontName='Times-Bold', fontSize=12, leading=16, alignment=1)))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("by", ParagraphStyle('TP12', fontName='Times-Roman', fontSize=10.5, leading=14, alignment=1)))
    story.append(Paragraph(f"<b>{student_name} ({roll_no})</b>", ParagraphStyle('TP13', fontName='Times-Bold', fontSize=12, leading=16, alignment=1)))
    story.append(Paragraph(f"Section: {section}, III Year, V Semester", ParagraphStyle('TP14', fontName='Times-Bold', fontSize=11, leading=15, alignment=1)))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("<b>Faculty:</b>", ParagraphStyle('TP15', fontName='Times-Bold', fontSize=11, leading=15, alignment=1)))
    story.append(Paragraph(f"<b>{faculty}</b>", ParagraphStyle('TP16', fontName='Times-Bold', fontSize=11.5, leading=15, alignment=1)))
    story.append(Paragraph("Assistant Professor", ParagraphStyle('TP17', fontName='Times-Roman', fontSize=10.5, leading=14, alignment=1)))
    story.append(Spacer(1, 0.35 * cm))

    # CBIT Shield Logo (exact dimensions matching reference)
    if os.path.exists('cbit_logo.png'):
        story.append(Image('cbit_logo.png', width=2.92 * cm, height=3.81 * cm))
        story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Department of Computer Science and Engineering,", ParagraphStyle('TP18', fontName='Times-Bold', fontSize=12, leading=16, alignment=1)))
    story.append(Paragraph("Chaitanya Bharathi Institute of Technology", ParagraphStyle('TP19', fontName='Times-Bold', fontSize=12.5, leading=17, alignment=1)))
    story.append(Paragraph("(Autonomous),", ParagraphStyle('TP20', fontName='Times-Bold', fontSize=11, leading=15, alignment=1)))
    story.append(Paragraph("(Affiliated to Osmania University, Hyderabad)", ParagraphStyle('TP21', fontName='Times-Roman', fontSize=10, leading=14, alignment=1)))
    story.append(Paragraph("Hyderabad, TELANGANA (INDIA) - 500 075", ParagraphStyle('TP22', fontName='Times-Bold', fontSize=10, leading=14, alignment=1)))
    story.append(Paragraph("[2026-2027]", ParagraphStyle('TP23', fontName='Times-Bold', fontSize=10.5, leading=15, alignment=1)))
    story.append(PageBreak())

    # ================= PAGE 2: ABSTRACT (ii) =================
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>Abstract</b>", ParagraphStyle('AbsTitle', fontName='Times-Bold', fontSize=14, leading=18, alignment=1)))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph(
        "This report presents two comprehensive case studies from the Fundamentals of Data Science course that apply "
        "unsupervised and supervised machine learning methodologies to real-world industrial datasets. "
        "Case Study 2 examines customer segmentation for East-West Airlines frequent flyer accounts (3,999 records). "
        "Continuous mileage features were normalized using log-transformations, and multivariate outliers were pruned using z-scores on transformed dimensions. "
        "Three distinct clustering algorithms (K-Means, Agglomerative Hierarchical clustering, and DBSCAN) were systematically fitted and evaluated. "
        "Inertia elbow and silhouette score analyses established an optimal four-cluster structure (silhouette score = 0.3199). "
        "The clusters clearly isolated distinct behavioral personas: Active Frequent Flyers (27.7%), Credit Card Loyalty Earners (45.0%), "
        "Inactive/Low Engagement Members (27.2%), and Elite Corporate Outliers (<0.1%), providing actionable marketing strategies.",
        body
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(
        "Case Study 5 investigates the predictive modeling of red Vinho Verde wine quality (Cortez et al., 2009) using laboratory-measured physicochemical attributes. "
        "Data hygiene checks identified and removed 240 duplicate records (15.01%) to prevent train-test data leakage. "
        "Descriptive statistics and relative dispersion analysis identified residual sugar as possessing the greatest relative variability (CV = 53.59%). "
        "A formal two-tailed hypothesis test verified a statistically significant moderate positive linear relationship between alcohol content and quality ratings (r = +0.4803, p = 2.28e-79). "
        "Multiple linear regression was trained on an 80:20 split with StandardScaler fitted strictly on training data. "
        "The model achieved test MAE = 0.5041, RMSE = 0.6565, and test R2 = 0.3915. "
        "Multicollinearity analysis via Variance Inflation Factor (VIF) highlighted interdependency between fixed acidity (VIF = 7.95) and density (VIF = 6.31). "
        "Standardized regression coefficients ranked alcohol (beta = +0.3340) and volatile acidity (beta = -0.1785) as primary quality drivers. "
        "Residual diagnostics revealed that target discreteness introduces parallel error bands and mean attenuation, leading to the practical conclusion that linear regression functions effectively as an automated vat pre-screening filter but cannot replace certified human sensory panels.",
        body
    ))
    story.append(PageBreak())

    # ================= PAGE 3: TABLE OF CONTENTS (iii) =================
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>Table of Contents</b>", ParagraphStyle('TOCTitle', fontName='Times-Bold', fontSize=14, leading=18, alignment=1)))
    story.append(Spacer(1, 0.4 * cm))

    def get_toc(key, default_val):
        return str(toc_pages.get(key, default_val)) if toc_pages else str(default_val)

    toc_data = [
        [Paragraph("<b>S.No</b>", tbl_header), Paragraph("<b>Title</b>", tbl_header), Paragraph("<b>Page</b>", tbl_header)],
        [Paragraph("", tbl_cell), Paragraph("Title Page", tbl_cell_left), Paragraph("i", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("Abstract", tbl_cell_left), Paragraph("ii", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("List of Figures", tbl_cell_left), Paragraph("iv", tbl_cell)],
        [Paragraph("<b>1.</b>", tbl_cell), Paragraph("<b>INTRODUCTION</b>", tbl_cell_left), Paragraph(f"<b>{get_toc('intro', 1)}</b>", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;1.1 Problem Definition including the significance and objective", tbl_cell_left), Paragraph(get_toc('p1_1', 1), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;1.2 Concepts used in Methodology", tbl_cell_left), Paragraph(get_toc('p1_2', 2), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;1.3 Outline the results", tbl_cell_left), Paragraph(get_toc('p1_3', 3), tbl_cell)],
        [Paragraph("<b>2.</b>", tbl_cell), Paragraph("<b>CASE STUDY 2: CLUSTERING ANALYSIS ON EAST-WEST AIRLINES</b>", tbl_cell_left), Paragraph(f"<b>{get_toc('cs2', 4)}</b>", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.1 Problem Statement", tbl_cell_left), Paragraph(get_toc('p2_1', 4), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.2 Dataset Description", tbl_cell_left), Paragraph(get_toc('p2_2', 4), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.3 Methodology", tbl_cell_left), Paragraph(get_toc('p2_3', 4), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.4 Statistical Findings", tbl_cell_left), Paragraph(get_toc('p2_4', 5), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.5 Model Evaluation", tbl_cell_left), Paragraph(get_toc('p2_5', 6), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.6 Interpretation and Business Recommendations", tbl_cell_left), Paragraph(get_toc('p2_6', 9), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;2.7 Conclusion and Limitations", tbl_cell_left), Paragraph(get_toc('p2_7', 9), tbl_cell)],
        [Paragraph("<b>3.</b>", tbl_cell), Paragraph("<b>CASE STUDY 5: WINE QUALITY PREDICTION VIA MULTIPLE REGRESSION</b>", tbl_cell_left), Paragraph(f"<b>{get_toc('cs5', 10)}</b>", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.1 Problem Statement", tbl_cell_left), Paragraph(get_toc('p3_1', 10), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.2 Dataset Description", tbl_cell_left), Paragraph(get_toc('p3_2', 10), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.3 Methodology", tbl_cell_left), Paragraph(get_toc('p3_3', 10), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.4 Statistical Findings", tbl_cell_left), Paragraph(get_toc('p3_4', 11), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.5 Model Evaluation", tbl_cell_left), Paragraph(get_toc('p3_5', 13), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.6 Interpretation and Business Recommendations", tbl_cell_left), Paragraph(get_toc('p3_6', 15), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;3.7 Conclusion and Limitations", tbl_cell_left), Paragraph(get_toc('p3_7', 16), tbl_cell)],
        [Paragraph("<b>4.</b>", tbl_cell), Paragraph("<b>CONCLUSION and References</b>", tbl_cell_left), Paragraph(f"<b>{get_toc('conc', 17)}</b>", tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;4.1 Conclusion", tbl_cell_left), Paragraph(get_toc('p4_1', 17), tbl_cell)],
        [Paragraph("", tbl_cell), Paragraph("&nbsp;&nbsp;4.2 References", tbl_cell_left), Paragraph(get_toc('p4_2', 17), tbl_cell)]
    ]
    t_toc = Table(toc_data, colWidths=[1.8*cm, 12.8*cm, 1.8*cm])
    t_toc.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # ================= PAGE 4: LIST OF FIGURES (iv) =================
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>List of Figures</b>", ParagraphStyle('LOFTitle', fontName='Times-Bold', fontSize=14, leading=18, alignment=1)))
    story.append(Spacer(1, 0.4 * cm))

    def get_lof(key, default_val):
        return str(lof_pages.get(key, default_val)) if lof_pages else str(default_val)

    lof_data = [
        [Paragraph("<b>Fig No.</b>", tbl_header), Paragraph("<b>Title</b>", tbl_header), Paragraph("<b>Page</b>", tbl_header)],
        [Paragraph("2.1", tbl_cell), Paragraph("Correlation heatmap of transformed customer attributes", tbl_cell_left), Paragraph(get_lof('fig2_1', 5), tbl_cell)],
        [Paragraph("2.2", tbl_cell), Paragraph("Initial 2D PCA projection of customer features", tbl_cell_left), Paragraph(get_lof('fig2_2', 5), tbl_cell)],
        [Paragraph("2.3", tbl_cell), Paragraph("K-Means optimal K selection via inertia elbow and silhouette score analysis", tbl_cell_left), Paragraph(get_lof('fig2_3', 6), tbl_cell)],
        [Paragraph("2.4", tbl_cell), Paragraph("Dendrograms for four hierarchical linkage criteria (Ward, Complete, Average, Single)", tbl_cell_left), Paragraph(get_lof('fig2_4', 6), tbl_cell)],
        [Paragraph("2.5", tbl_cell), Paragraph("5-Nearest neighbor distance plot for DBSCAN eps parameter selection", tbl_cell_left), Paragraph(get_lof('fig2_5', 7), tbl_cell)],
        [Paragraph("2.6", tbl_cell), Paragraph("2D PCA cluster visualizations comparing K-Means, Hierarchical (Ward), and DBSCAN", tbl_cell_left), Paragraph(get_lof('fig2_6', 8), tbl_cell)],
        [Paragraph("2.7", tbl_cell), Paragraph("Standardized feature profile heatmap across customer segments", tbl_cell_left), Paragraph(get_lof('fig2_7', 9), tbl_cell)],
        [Paragraph("3.1", tbl_cell), Paragraph("Red wine sensory quality rating distribution showing class concentration", tbl_cell_left), Paragraph(get_lof('fig3_1', 11), tbl_cell)],
        [Paragraph("3.2", tbl_cell), Paragraph("Alcohol content grouped by quality score boxplot showing median trend", tbl_cell_left), Paragraph(get_lof('fig3_2', 12), tbl_cell)],
        [Paragraph("3.3", tbl_cell), Paragraph("Correlation heatmap of physicochemical attributes and quality", tbl_cell_left), Paragraph(get_lof('fig3_3', 12), tbl_cell)],
        [Paragraph("3.4", tbl_cell), Paragraph("Standardized regression coefficients ranked by absolute impact |β|", tbl_cell_left), Paragraph(get_lof('fig3_4', 13), tbl_cell)],
        [Paragraph("3.5", tbl_cell), Paragraph("Predicted vs actual quality and residual diagnostics showing discrete score stripes", tbl_cell_left), Paragraph(get_lof('fig3_5', 14), tbl_cell)]
    ]
    t_lof = Table(lof_data, colWidths=[1.8*cm, 12.8*cm, 1.8*cm])
    t_lof.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_lof)
    story.append(PageBreak())

    # ================= CHAPTER 1: INTRODUCTION (ARABIC PAGE 1) =================
    story.append(Paragraph("1. INTRODUCTION", ch_title))
    story.append(Paragraph(
        "This report covers two case studies from the Fundamentals of Data Science course. "
        "The first is an unsupervised clustering problem aimed at customer segmentation on commercial airline loyalty data, "
        "and the second is a supervised multiple linear regression problem predicting wine sensory ratings from laboratory physicochemical measurements. "
        "Both case studies follow the rigorous data science lifecycle: data understanding, hygiene and preprocessing, exploratory data analysis, "
        "model construction and hyperparameter optimization, diagnostic evaluation on unseen data, and actionable business interpretation.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("1.1 Problem Definition including the significance and objective", sec_title))
    story.append(Paragraph("Problem Definition", subsec_title))
    story.append(Paragraph(
        "In <b>Case Study 2</b>, East-West Airlines has accumulated flight activity, mileage accrual, credit card partnership transactions, "
        "and enrollment tenure for 3,999 frequent flyer accounts. The problem is that customers exhibit highly diverse engagement behaviors, "
        "rendering uniform marketing promotions inefficient and expensive. The task is to discover cohesive, natural customer segments "
        "using unsupervised clustering algorithms (K-Means, Hierarchical, DBSCAN) so that marketing campaigns can be tailored to distinct loyalty profiles.<br/>"
        "In <b>Case Study 5</b>, wine sensory quality evaluation traditionally relies on certified human tasting panels. This process is expensive, "
        "time-consuming, and subject to expert palate fatigue and individual subjectivity. The problem is to determine whether objective, "
        "laboratory-measured physicochemical attributes (such as acidity, sulfur dioxide, density, pH, and alcohol) can reliably model and predict "
        "sensory quality scores using inferential statistics and multiple linear regression.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("Significance", subsec_title))
    story.append(Paragraph("• Customer segmentation enables airlines to allocate loyalty rewards efficiently, reserving premium perks for high-flight travelers while incentivizing credit card spenders to redeem accumulated miles for flights.", bullet_style))
    story.append(Paragraph("• Demonstrates how proper data transformation (log1p) and principled z-score outlier filtering prevent extreme values from distorting Euclidean distance metrics in unsupervised learning.", bullet_style))
    story.append(Paragraph("• In viticulture, an objective statistical model allows wineries to automate vat pre-screening during fermentation, identifying chemically flawed batches before submission to human panels.", bullet_style))
    story.append(Paragraph("• Demonstrates how regression diagnostics (multicollinearity via VIF and residual analysis) uncover model limitations when predicting discrete ordinal targets.", bullet_style))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("Objectives", subsec_title))
    story.append(Paragraph("1. To conduct exploratory data analysis with descriptive statistics, distribution histograms, and correlation heatmaps across both datasets.", bullet_style))
    story.append(Paragraph("2. To handle data hygiene challenges, including right-skewness and multivariate outliers in Case Study 2, and deduplication in Case Study 5 without leaking test information.", bullet_style))
    story.append(Paragraph("3. To evaluate and compare K-Means, Agglomerative Hierarchical clustering, and DBSCAN on customer segmentation using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.", bullet_style))
    story.append(Paragraph("4. To build, evaluate, and diagnose a multiple linear regression model on wine physicochemical attributes using MAE, MSE, RMSE, R2, and VIF multicollinearity checks.", bullet_style))
    story.append(Paragraph("5. To formulate evidence-backed business recommendations regarding customer marketing strategies and the feasibility of replacing human sensory panels.", bullet_style))
    story.append(PageBreak())

    # ================= CHAPTER 1.2: CONCEPTS USED IN METHODOLOGY (ARABIC PAGE 2) =================
    story.append(Paragraph("1.2 Concepts used in Methodology", sec_title))
    story.append(Paragraph("Data Preparation Concepts", subsec_title))
    story.append(Paragraph("• <b>Exploratory Data Analysis (EDA):</b> Examination of central tendency, dispersion, skewness, and inter-feature correlation through visualizations to uncover underlying structural properties.", bullet_style))
    story.append(Paragraph("• <b>Logarithmic Transformation (log1p):</b> Continuous financial and mileage variables with severe right-skewness (skew > 1.0) are transformed using log(1 + x) to compress long tails and stabilize variance.", bullet_style))
    story.append(Paragraph("• <b>Multivariate Outlier Treatment:</b> Standard z-score filtering (|z| > 3) applied to log-transformed dimensions removes mathematical anomalies while preserving legitimate high-tier frequent flyers.", bullet_style))
    story.append(Paragraph("• <b>Data Hygiene & Deduplication:</b> Identical duplicate records are dropped prior to modeling to avoid data leakage and prevent artificially optimistic evaluation metrics.", bullet_style))
    story.append(Paragraph("• <b>Feature Standardization:</b> StandardScaler transforms features to mean = 0 and std = 1. In regression, the scaler is fitted strictly on training data to prevent test leakage.", bullet_style))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("Machine Learning & Clustering Concepts", subsec_title))
    story.append(Paragraph("• <b>K-Means Clustering:</b> Partitions observations into K clusters by minimizing Within-Cluster Sum of Squares (WCSS / inertia) using Lloyd's algorithm. Optimal K is determined by balancing the inertia elbow with silhouette scores.", bullet_style))
    story.append(Paragraph("• <b>Hierarchical Agglomerative Clustering:</b> Builds a bottom-up dendrogram using pairwise Euclidean distances. Evaluates Ward's minimum variance, Complete, Average, and Single linkage criteria.", bullet_style))
    story.append(Paragraph("• <b>DBSCAN:</b> Density-Based Spatial Clustering of Applications with Noise groups points exceeding min_samples within an eps radius, separating dense core structures from sparse noise.", bullet_style))
    story.append(Paragraph("• <b>Multiple Linear Regression:</b> Models the linear relationship between continuous physicochemical inputs and sensory quality via Ordinary Least Squares (OLS).", bullet_style))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("Evaluation Concepts", subsec_title))
    story.append(Paragraph("• <b>Silhouette Score:</b> Measures how cohesive a point is to its own cluster compared to neighboring clusters (-1 to +1; higher is better).", bullet_style))
    story.append(Paragraph("• <b>Davies-Bouldin Index:</b> Average similarity ratio of each cluster with its most similar cluster (lower is better; 0 is ideal).", bullet_style))
    story.append(Paragraph("• <b>Calinski-Harabasz Score:</b> Variance ratio criterion evaluating between-cluster dispersion relative to within-cluster dispersion (higher is better).", bullet_style))
    story.append(Paragraph("• <b>Regression Metrics:</b> MAE (average absolute error in rating points), MSE (penalizes large errors quadratically), RMSE (root mean squared error in target units), and R2 (proportion of variance explained).", bullet_style))
    story.append(Paragraph("• <b>Variance Inflation Factor (VIF):</b> Quantifies coefficient variance inflation due to collinearity (VIF > 5 signals moderate-to-high multicollinearity).", bullet_style))
    story.append(PageBreak())

    # ================= CHAPTER 1.3: OUTLINE THE RESULTS (ARABIC PAGE 3) =================
    story.append(Paragraph("1.3 Outline the results", sec_title))
    story.append(Paragraph("Chapter 2 presents Case Study 2 and Chapter 3 presents Case Study 5. The primary empirical findings include:", body))
    story.append(Paragraph("• <b>Case Study 2:</b> K-Means and Hierarchical Ward linkage identified optimal partitions at K = 4 (Silhouette = 0.3199 and 0.3317, respectively). DBSCAN separated 3 core clusters while isolating 220 noise accounts (6.07%).", bullet_style))
    story.append(Paragraph("• Four distinct airline personas were uncovered: Active Frequent Flyers (27.7%, 1,076 flight mi/yr, 54.2% award rate), Credit Card Loyalty Earners (45.0%, 20,900 bonus mi, 0 flight mi), Inactive Members (27.2%), and Outlier Accounts (<0.1%).", bullet_style))
    story.append(Paragraph("• <b>Case Study 5:</b> Multiple linear regression achieved test MAE = 0.5041, RMSE = 0.6565, and test R2 = 0.3915 (train R2 = 0.3520), confirming stable out-of-sample generalization without overfitting.", bullet_style))
    story.append(Paragraph("• Alcohol content (beta = +0.3340, p < 0.0001) and Volatile Acidity (beta = -0.1785, p < 0.0001) emerged as the two primary chemical determinants of sensory preference.", bullet_style))
    story.append(Paragraph("• Residual diagnostics demonstrated that target discreteness introduces parallel error bands and mean attenuation, confirming that the model should function as an automated vat pre-screening tool rather than a sensory panel replacement.", bullet_style))
    story.append(PageBreak())

    # ================= CHAPTER 2: CASE STUDY 2 (ARABIC PAGE 4) =================
    story.append(Paragraph("2. CASE STUDY 2: CLUSTERING ANALYSIS ON EAST-WEST AIRLINES", ch_title))
    story.append(Paragraph("2.1 Problem Statement", sec_title))
    story.append(Paragraph(
        "East-West Airlines seeks to optimize its frequent flyer loyalty program by identifying meaningful customer segments "
        "from operational records of 3,999 accounts. Customers accrue mileage through paid flights, partner airline activity, "
        "and co-branded credit card transactions. Applying identical marketing promotions across all members causes inefficient capital allocation: "
        "frequent business travelers receive offers for credit card points they do not need, while credit card spenders receive flight promotions they rarely use. "
        "The objective is to apply unsupervised clustering techniques to discover natural behavioral groupings, evaluate cluster cohesion, "
        "and establish targeted marketing recommendations for each segment.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("2.2 Dataset Description", sec_title))
    story.append(Paragraph(
        "The dataset was compiled by Shmueli and Bruce (2016) and contains 3,999 observations across 12 features with zero missing values. "
        "The features capture accumulated balance, qualifying miles, credit card mileage bins (cc1_miles, cc2_miles, cc3_miles), "
        "bonus miles and transactions, flight miles and transactions over the past 12 months, days since enrollment, and an award redemption indicator. "
        "Table 2.1 summarizes the key continuous numerical features.",
        body
    ))

    # Table 2.1: Summary Stats Table (Key 6 features)
    story.append(Paragraph("<b>Table 2.1: Summary statistics of East-West Airlines features</b>", table_caption))
    tbl2_rows = [
        [Paragraph("<b>Feature</b>", tbl_header), Paragraph("<b>Count</b>", tbl_header), Paragraph("<b>Mean</b>", tbl_header),
         Paragraph("<b>Std</b>", tbl_header), Paragraph("<b>Min</b>", tbl_header), Paragraph("<b>Median</b>", tbl_header), Paragraph("<b>Max</b>", tbl_header)]
    ]
    key_cs2_feats = ['Balance', 'Qual_miles', 'cc1_miles', 'Bonus_miles', 'Flight_miles_12mo', 'Days_since_enroll']
    cs2_stats_sub = cs2_stats_df[cs2_stats_df['Feature'].isin(key_cs2_feats)]
    for _, row in cs2_stats_sub.iterrows():
        tbl2_rows.append([
            Paragraph(str(row['Feature']), tbl_cell_left),
            Paragraph(str(row['Count']), tbl_cell),
            Paragraph(f"{row['Mean']:.2f}", tbl_cell),
            Paragraph(f"{row['Std']:.2f}", tbl_cell),
            Paragraph(f"{row['Min']:.2f}", tbl_cell),
            Paragraph(f"{row['Median']:.2f}", tbl_cell),
            Paragraph(f"{row['Max']:.2f}", tbl_cell)
        ])
    t_cs2_stats = Table(tbl2_rows, colWidths=[3.8*cm, 1.6*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.2*cm, 2.4*cm])
    t_cs2_stats.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_cs2_stats)
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("2.3 Methodology", sec_title))
    story.append(Paragraph(
        "The customer segmentation workflow is executed through a six-stage unsupervised learning pipeline:",
        body
    ))
    story.append(Paragraph(
        "• <b>Feature Selection & Data Hygiene:</b> The arbitrary record identifier ID# was dropped as it contains no behavioral information. The binary label Award? was set aside from clustering to serve as an external validation variable. The dataset was verified to have zero missing values across all 3,999 observations.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Skewness Diagnosis & Log Transformation:</b> Continuous mileage features showed extreme right-skewness (skewness up to +7.45) with multi-order-of-magnitude ranges. The natural log transformation log1p(x) = log(1 + x) was applied to compress long tails, stabilize feature variance, and normalize distributions.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Multivariate Outlier Treatment:</b> Standard z-score filtering (|z| > 3) applied to log-transformed dimensions identified and pruned 376 extreme multivariate outliers (9.40%), retaining 3,623 robust records. This prevents distance distortions while keeping legitimate high-tier frequent flyers.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Feature Standardization:</b> StandardScaler was applied to all 10 feature dimensions to achieve mean = 0 and variance = 1, ensuring equal mathematical weighting during Euclidean distance computations across categorical bins and continuous mileage metrics.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Clustering Algorithms & Tuning:</b> Three clustering paradigms were evaluated: K-Means (tested for K = 1 to 10 with Lloyd's algorithm, analyzing inertia elbow and silhouette profiles); Agglomerative Hierarchical clustering (evaluating Ward, Complete, Average, and Single linkage criteria); and DBSCAN (calibrating eps via 5-nearest neighbor k-distance knee analysis).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Segment Profiling & Validation:</b> Discovered segments were projected onto 2D PCA space, profiled using unscaled original feature means, and cross-referenced against historical award redemption rates to design targeted marketing strategies.",
        bullet_style
    ))
    story.append(PageBreak())

    # ================= CS-2 STATISTICAL FINDINGS (ARABIC PAGE 5) =================
    story.append(Paragraph("2.4 Statistical Findings", sec_title))
    story.append(Paragraph(
        "Distribution analysis of continuous customer mileage features indicates extreme positive skewness in raw data, with severe concentration near zero and thin tails extending over millions of miles. Natural log transformation log1p(x) = log(1 + x) compressed skewness significantly across all metrics: Balance skewness dropped from +5.00 to -0.43, Bonus_miles from +2.84 to -1.43, Bonus_trans from +1.16 to -0.78, and Flight_miles_12mo from +7.45 to +1.05.",
        body
    ))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Figure 2.1 presents the correlation heatmap across transformed customer features. "
        "Strong positive correlations exist between credit card tier cc1_miles and bonus miles (r = +0.83) as well as bonus transactions (r = +0.60). "
        "Flight miles correlate strongly with flight transactions (r = +0.82) but exhibit near-zero correlation with credit card mileage (r = +0.06), "
        "confirming that credit card reward collectors and frequent business travelers represent fundamentally distinct operational customer cohorts.",
        body
    ))

    # Figure 2.1: Correlation Heatmap
    if os.path.exists('figures/cs2_03_correlation_heatmap.png'):
        story.append(Image('figures/cs2_03_correlation_heatmap.png', width=11.5*cm, height=5.0*cm))
        story.append(Paragraph("Fig 2.1: Correlation heatmap of transformed customer attributes", fig_caption))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Figure 2.2 illustrates the initial 2D PCA projection of the standardized customer feature space. "
        "The first two principal components explain 40.2% of total variance (PC1: 28.2%, PC2: 12.0%). "
        "The projection reveals a dense central mass of casual flyers, flanked by extended projections representing heavy credit card spenders and frequent flyers.",
        body
    ))

    # Figure 2.2: Initial PCA
    if os.path.exists('figures/cs2_05_pca_initial.png'):
        story.append(Image('figures/cs2_05_pca_initial.png', width=11.5*cm, height=4.6*cm))
        story.append(Paragraph("Fig 2.2: Initial 2D PCA projection of customer features", fig_caption))

    story.append(PageBreak())

    # ================= CS-2 2.5 MODEL EVALUATION (ARABIC PAGE 6) =================
    story.append(Paragraph("2.5 Model Evaluation", sec_title))
    story.append(Paragraph(
        "Three clustering algorithms were evaluated across standardized dimensions:<br/>"
        "1. <b>K-Means:</b> Evaluated across K = 1 to 10 (Figure 2.3). Inertia (WCSS) drops steeply from K = 1 to K = 4 before leveling off into an elbow knee. "
        "The silhouette score profile peaks at K = 4 with a score of 0.3199 (Inertia = 12,568), establishing K = 4 as the mathematically and operationally optimal choice.",
        body
    ))

    # Figure 2.3: K-Means Elbow & Silhouette
    if os.path.exists('figures/cs2_06_kmeans_elbow_silhouette.png'):
        story.append(Image('figures/cs2_06_kmeans_elbow_silhouette.png', width=12.5*cm, height=4.3*cm))
        story.append(Paragraph("Fig 2.3: K-Means optimal K selection via inertia elbow and silhouette score analysis", fig_caption))

    story.append(Paragraph(
        "2. <b>Hierarchical Clustering:</b> Dendrograms were constructed for Ward, Complete, Average, and Single linkage (Figure 2.4). "
        "Ward linkage produced the highest silhouette score (0.3317 at K = 4) and well-balanced clusters. Complete linkage achieved 0.2453, Average achieved 0.2518, "
        "and Single linkage suffered from severe chaining (silhouette = 0.2641).",
        body
    ))

    # Figure 2.4: Dendrograms
    if os.path.exists('figures/cs2_07_dendrograms.png'):
        story.append(Image('figures/cs2_07_dendrograms.png', width=12.5*cm, height=4.5*cm))
        story.append(Paragraph("Fig 2.4: Dendrograms for four hierarchical linkage criteria (Ward, Complete, Average, Single)", fig_caption))

    story.append(PageBreak())

    # ================= CS-2 DENDROGRAMS, DBSCAN, & COMPARISON (ARABIC PAGE 7) =================
    story.append(Paragraph(
        "3. <b>DBSCAN:</b> Evaluated using 5-nearest neighbor distance analysis (Figure 2.5). "
        "A sharp knee appears around distance 0.88, beyond which points represent isolated noise. "
        "Fitting DBSCAN with optimal parameters eps = 0.879 and min_samples = 10 separated 3 dense clusters while isolating 220 noise points (6.07%). "
        "Silhouette score on non-noise core points was 0.3353. However, DBSCAN placed 93.9% of members into a single giant cluster due to uniform density in the central core.",
        body
    ))

    # Figure 2.5: DBSCAN Knee Plot
    if os.path.exists('figures/cs2_08_kdistance_plot.png'):
        story.append(Image('figures/cs2_08_kdistance_plot.png', width=11.0*cm, height=4.2*cm))
        story.append(Paragraph("Fig 2.5: 5-Nearest neighbor distance plot for DBSCAN eps parameter selection", fig_caption))

    story.append(Spacer(1, 0.15 * cm))

    # Table 2.2: Algorithm Comparison Table
    story.append(Paragraph("<b>Table 2.2: Clustering Algorithm Evaluation Comparison</b>", table_caption))
    tbl_comp_rows = [
        [Paragraph("<b>Algorithm</b>", tbl_header), Paragraph("<b>Clusters</b>", tbl_header),
         Paragraph("<b>Silhouette</b>", tbl_header), Paragraph("<b>Davies-Bouldin</b>", tbl_header),
         Paragraph("<b>Calinski-Harabasz</b>", tbl_header)]
    ]
    for _, row in cs2_comp_df.iterrows():
        tbl_comp_rows.append([
            Paragraph(str(row['Algorithm']), tbl_cell_left),
            Paragraph(str(row['Clusters']), tbl_cell),
            Paragraph(f"{row['Silhouette']:.4f}", tbl_cell),
            Paragraph(f"{row['Davies-Bouldin']:.4f}", tbl_cell),
            Paragraph(f"{row['Calinski-Harabasz']:.2f}", tbl_cell)
        ])
    t_comp = Table(tbl_comp_rows, colWidths=[4.2*cm, 2.0*cm, 2.8*cm, 3.2*cm, 3.8*cm])
    t_comp.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "<b>Synthesis of Clustering Results:</b> While Ward hierarchical clustering achieved the highest silhouette score (0.3317), "
        "K-Means (K = 4) demonstrated comparable cluster separation (silhouette = 0.3199, Calinski-Harabasz = 1512.32) with superior computational efficiency "
        "and easily deployable centroids for scoring new incoming customer accounts. DBSCAN proved unviable for operational customer tiering due to heavy cluster skewness.",
        body
    ))
    story.append(PageBreak())

    # ================= CS-2 CLUSTER PROJECTIONS & PROFILES (ARABIC PAGE 8) =================
    story.append(Paragraph(
        "Figure 2.6 compares 2D PCA cluster scatter plots across K-Means, Hierarchical Ward, and DBSCAN. "
        "K-Means and Ward linkage produce nearly identical customer boundaries, cleanly separating credit card spenders (top) from frequent flyers (bottom). "
        "DBSCAN flags sparse peripheral flyers as noise while merging the remainder into a single cluster.",
        body
    ))

    # Figure 2.6: PCA Cluster Scatters
    if os.path.exists('figures/cs2_09_cluster_pca_scatter.png'):
        story.append(Image('figures/cs2_09_cluster_pca_scatter.png', width=14.5*cm, height=5.0*cm))
        story.append(Paragraph("Fig 2.6: 2D PCA cluster visualizations comparing K-Means, Hierarchical (Ward), and DBSCAN", fig_caption))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Table 2.3 presents unscaled mean feature values and Award redemption rates per K-Means cluster.",
        body
    ))

    # Table 2.3: Cluster Profiles Table
    story.append(Paragraph("<b>Table 2.3: Mean unscaled feature values and award redemption rates by cluster</b>", table_caption))
    tbl_prof_rows = [
        [Paragraph("<b>Cluster</b>", tbl_header), Paragraph("<b>Count</b>", tbl_header), Paragraph("<b>Share</b>", tbl_header),
         Paragraph("<b>Balance</b>", tbl_header), Paragraph("<b>Bonus Mi</b>", tbl_header), Paragraph("<b>Flt Mi 12m</b>", tbl_header),
         Paragraph("<b>Days Enroll</b>", tbl_header), Paragraph("<b>Award Rate</b>", tbl_header)]
    ]
    for _, row in cs2_prof_df.iterrows():
        tbl_prof_rows.append([
            Paragraph(f"Cluster {int(row['Cluster'])}", tbl_cell_left),
            Paragraph(str(int(row['Customer_Count'])), tbl_cell),
            Paragraph(f"{row['Percentage(%)']:.1f}%", tbl_cell),
            Paragraph(f"{row['Balance']:,.0f}", tbl_cell),
            Paragraph(f"{row['Bonus_miles']:,.0f}", tbl_cell),
            Paragraph(f"{row['Flight_miles_12mo']:,.0f}", tbl_cell),
            Paragraph(f"{row['Days_since_enroll']:,.0f}", tbl_cell),
            Paragraph(f"{row['Award?']*100:.1f}%", tbl_cell)
        ])
    t_prof = Table(tbl_prof_rows, colWidths=[2.2*cm, 1.6*cm, 1.6*cm, 2.5*cm, 2.4*cm, 2.4*cm, 2.5*cm, 2.2*cm])
    t_prof.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_prof)
    story.append(PageBreak())

    # ================= CS-2 HEATMAP, PERSONAS & CONCLUSION (ARABIC PAGE 9) =================
    # Figure 2.7: Cluster Profile Heatmap
    if os.path.exists('figures/cs2_10_cluster_heatmap.png'):
        story.append(Image('figures/cs2_10_cluster_heatmap.png', width=13.0*cm, height=4.8*cm))
        story.append(Paragraph("Fig 2.7: Standardized feature profile heatmap across customer segments", fig_caption))

    story.append(Paragraph("2.6 Interpretation and Business Recommendations", sec_title))
    story.append(Paragraph("Customer Segment Personas & Marketing Recommendations", subsec_title))
    story.append(Paragraph(
        "1. <b>Cluster 0: Active Frequent Flyers (27.7% of accounts):</b> Characterized by high flight activity (1,076 flight miles/year, 3.2 flight transactions) "
        "and the highest Award flight redemption rate (54.2%). These represent core business travelers who generate premium seat revenue. "
        "<i>Marketing Strategy:</i> Retain tier status with complimentary airport lounge access, priority check-in/boarding, seat upgrade certificates, and elite mileage multipliers.",
        body
    ))
    story.append(Paragraph(
        "2. <b>Cluster 1: Credit Card Loyalty Earners (45.0% of accounts):</b> The largest segment, characterized by high credit card spend (20,900 bonus miles, 15.5 transactions) "
        "but near-zero actual flight activity (0 flight miles/year). These members accumulate points through partner retail programs rather than travel. "
        "<i>Marketing Strategy:</i> Implement flight conversion promotions (e.g., 20% discount on flight award redemptions or co-branded hotel/flight packages) to convert card spenders into active airline travelers.",
        body
    ))
    story.append(Paragraph(
        "3. <b>Cluster 2: Inactive / Low Engagement Members (27.2% of accounts):</b> Characterized by low balances (22,418 miles), minimal transactions, and low award redemption rate (11.8%). "
        "These represent dormant or casual flyers at risk of churn. "
        "<i>Marketing Strategy:</i> Launch automated re-activation campaigns, milestone bonus challenges (e.g., 'Earn 2,000 bonus miles on your next booking'), and expiration warning notices.",
        body
    ))
    story.append(Paragraph(
        "4. <b>Cluster 3: Elite Outlier Accounts (<0.1%):</b> High-volume qualifying accounts requiring dedicated corporate concierge support.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("2.7 Conclusion and Limitations", sec_title))
    story.append(Paragraph(
        "Clustering analysis successfully segmented East-West Airlines customers into actionable operational tiers. "
        "K-Means (K = 4) and Hierarchical Ward linkage provided clear, interpretable partitions that align with airline business reality. "
        "Key limitations include: (1) cluster count selection involves mathematical trade-offs between inertia minimization and business interpretability; "
        "(2) DBSCAN struggled with non-uniform density, grouping 94% of members into a single broad cluster; "
        "(3) PCA 2D projections retain only ~40% of variance, compressing high-dimensional cluster geometry; and "
        "(4) results depend on log-transformation thresholds and outlier cleaning parameters.",
        body
    ))
    story.append(PageBreak())

    # ================= CHAPTER 3: CASE STUDY 5 (ARABIC PAGE 10) =================
    story.append(Paragraph("3. CASE STUDY 5: WINE QUALITY PREDICTION VIA MULTIPLE REGRESSION", ch_title))
    story.append(Paragraph("3.1 Problem Statement", sec_title))
    story.append(Paragraph(
        "Wine quality assessment traditionally relies on certified human tasting panels. "
        "While authoritative, sensory panels are expensive, time-consuming, and subject to taster fatigue and subjective sensory bias. "
        "The objective of this case study is to determine whether objective laboratory-measured physicochemical properties of red wine "
        "can predict sensory quality scores assigned by expert panels. "
        "Using inferential statistical hypothesis testing and multiple linear regression, we seek to identify which chemical attributes "
        "most strongly influence sensory perception, evaluate model predictive accuracy, diagnose multicollinearity, and assess the feasibility "
        "of deploying an automated screening system in winery operations.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("3.2 Dataset Description", sec_title))
    story.append(Paragraph(
        "The analysis is conducted on the red wine variant of the Portuguese Vinho Verde dataset compiled by Cortez et al. (2009). "
        "The dataset contains 1,599 observations across 11 continuous physicochemical inputs and 1 integer quality target (median score on a 0 to 10 scale). "
        "Data hygiene checks identified zero missing values and 240 duplicate rows (15.01%). "
        "Duplicate rows were dropped prior to model training to prevent data leakage across train/test partitions, yielding 1,359 clean, unique wine samples. "
        "Table 3.1 summarizes the descriptive statistics for 6 key physicochemical attributes and the quality target.",
        body
    ))

    # Table 3.1: Red Wine Summary Stats Table (Key 6 features)
    story.append(Paragraph("<b>Table 3.1: Summary statistics of Red Wine physicochemical features</b>", table_caption))
    tbl3_rows = [
        [Paragraph("<b>Feature</b>", tbl_header), Paragraph("<b>Count</b>", tbl_header), Paragraph("<b>Mean</b>", tbl_header),
         Paragraph("<b>Std</b>", tbl_header), Paragraph("<b>Min</b>", tbl_header), Paragraph("<b>Median</b>", tbl_header), Paragraph("<b>Max</b>", tbl_header)]
    ]
    key_cs5_feats = ['fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides', 'alcohol', 'quality']
    cs5_stats_sub = cs5_stats_full_df[cs5_stats_full_df['Feature'].isin(key_cs5_feats)]
    for _, row in cs5_stats_sub.iterrows():
        tbl3_rows.append([
            Paragraph(str(row['Feature']), tbl_cell_left),
            Paragraph(str(row['Count']), tbl_cell),
            Paragraph(f"{row['Mean']:.2f}", tbl_cell),
            Paragraph(f"{row['Std']:.2f}", tbl_cell),
            Paragraph(f"{row['Min']:.2f}", tbl_cell),
            Paragraph(f"{row['Median']:.2f}", tbl_cell),
            Paragraph(f"{row['Max']:.2f}", tbl_cell)
        ])
    t_cs5_stats = Table(tbl3_rows, colWidths=[3.8*cm, 1.6*cm, 2.2*cm, 2.2*cm, 2.0*cm, 2.2*cm, 2.4*cm])
    t_cs5_stats.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_cs5_stats)
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("3.3 Methodology", sec_title))
    story.append(Paragraph(
        "The regression modeling and diagnostic evaluation follows a seven-stage statistical learning procedure:",
        body
    ))
    story.append(Paragraph(
        "• <b>Data Hygiene & Deduplication:</b> Integrity screening verified zero missing values. An exhaustive scan identified 240 identical duplicate records (15.01%), which were pruned prior to model training to prevent data leakage across partitions, leaving 1,359 unique samples.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Train-Test Partitioning:</b> The clean dataset was partitioned into an 80% training set (1,087 samples) and a 20% holdout test set (272 samples) with a fixed seed (random_state=42) before any modeling or parameter estimation.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Leakage-Free Feature Scaling:</b> StandardScaler was fitted strictly on the 1,087 training rows. The learned training means and standard deviations were then applied to scale both train and test partitions, preventing out-of-sample data leakage.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Exploratory Data & Dispersion Analysis:</b> Central tendency and dispersion metrics were calculated. Comparative relative variability was quantified using the Coefficient of Variation (CV = (std / mean) * 100) across key chemical attributes.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Inferential Hypothesis Testing:</b> A formal two-tailed Pearson correlation hypothesis test was conducted between alcohol content and sensory quality (H0: r = 0 vs H1: r != 0, alpha = 0.05) using Student's t-distribution.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Multiple Linear Regression:</b> An Ordinary Least Squares (OLS) model was fitted to predict wine sensory quality from all 11 physicochemical predictors, minimizing the sum of squared errors.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Model Diagnostics & Validation:</b> The model was evaluated on the holdout test set (MAE, MSE, RMSE, R2), tested for stability via 5-fold cross-validation, screened for multicollinearity using Variance Inflation Factor (VIF), and analyzed for residual patterns.",
        bullet_style
    ))
    story.append(PageBreak())

    # ================= CS-5 STATISTICAL FINDINGS (ARABIC PAGE 11) =================
    story.append(Paragraph("3.4 Statistical Findings", sec_title))
    story.append(Paragraph(
        "<b>Task 1a: Descriptive Statistics and Skewness:</b> "
        "Descriptive metrics were computed for alcohol, volatile acidity, and residual sugar (Table 3.2). "
        "Alcohol exhibited moderate positive skewness (+0.86) with mean 10.43% and median 10.20%. "
        "Volatile acidity showed moderate positive skewness (+0.73) with mean 0.53 g/dm3 and median 0.52. "
        "Residual sugar displayed extreme positive skewness (+4.55), reflecting that the majority of Vinho Verde wines are dry "
        "(median 2.20 g/dm3, mode 2.00 g/dm3), with a sparse tail of sweet dessert-style outliers reaching 15.5 g/dm3.",
        body
    ))

    # Table 3.2: Task 1a Stats
    story.append(Paragraph("<b>Table 3.2: Descriptive statistics and skewness for key attributes</b>", table_caption))
    tbl_t1_rows = [
        [Paragraph("<b>Attribute</b>", tbl_header), Paragraph("<b>Mean</b>", tbl_header), Paragraph("<b>Median</b>", tbl_header),
         Paragraph("<b>Mode</b>", tbl_header), Paragraph("<b>Std Dev</b>", tbl_header), Paragraph("<b>Skewness</b>", tbl_header)]
    ]
    for _, row in cs5_stats_df.iterrows():
        tbl_t1_rows.append([
            Paragraph(str(row['Attribute']), tbl_cell_left),
            Paragraph(f"{row['Mean']:.4f}", tbl_cell),
            Paragraph(f"{row['Median']:.2f}", tbl_cell),
            Paragraph(f"{row['Mode']:.2f}", tbl_cell),
            Paragraph(f"{row['Std_Dev']:.4f}", tbl_cell),
            Paragraph(f"{row['Skewness']:.4f}", tbl_cell)
        ])
    t_t1 = Table(tbl_t1_rows, colWidths=[3.8*cm, 2.4*cm, 2.2*cm, 2.2*cm, 2.8*cm, 2.8*cm])
    t_t1.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_t1)
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph(
        "<b>Task 1b: Relative Dispersion via Coefficient of Variation (CV):</b> "
        "Table 3.3 presents the Coefficient of Variation (CV = (std / mean) * 100) across the three key attributes. "
        "Residual sugar exhibits the greatest relative variability (CV = 53.59%), followed by Volatile Acidity (34.57%) and Alcohol (10.37%). "
        "CV is the mathematically appropriate metric for comparative dispersion because standard deviation depends on physical units (% vol vs g/dm3), "
        "whereas CV normalizes variability relative to the feature mean.",
        body
    ))

    # Table 3.3: Task 1b CV Table
    story.append(Paragraph("<b>Table 3.3: Coefficient of Variation (CV) comparison</b>", table_caption))
    tbl_cv_rows = [
        [Paragraph("<b>Attribute</b>", tbl_header), Paragraph("<b>Mean</b>", tbl_header),
         Paragraph("<b>Std Dev</b>", tbl_header), Paragraph("<b>CV(%)</b>", tbl_header)]
    ]
    for _, row in cs5_cv_df.iterrows():
        tbl_cv_rows.append([
            Paragraph(str(row['Attribute']), tbl_cell_left),
            Paragraph(f"{row['Mean']:.4f}", tbl_cell),
            Paragraph(f"{row['Std_Dev']:.4f}", tbl_cell),
            Paragraph(f"{row['CV(%)']:.2f}%", tbl_cell)
        ])
    t_cv = Table(tbl_cv_rows, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    t_cv.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_cv)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "<b>Task 1c: Pearson Correlation Hypothesis Test (Alcohol vs Quality):</b><br/>"
        "A formal two-tailed hypothesis test was executed (H0: r = 0 vs H1: r != 0, alpha = 0.05). "
        "The sample Pearson correlation was r = +0.4803 with t = 20.34 and p-value = 2.28e-79. "
        "Since p < 0.05, we reject the null hypothesis and conclude that a statistically significant, moderate positive linear relationship exists.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    # Figure 3.2: Quality Distribution
    if os.path.exists('figures/cs5_01_quality_distribution.png'):
        story.append(Image('figures/cs5_01_quality_distribution.png', width=12.0*cm, height=4.6*cm))
        story.append(Paragraph("Fig 3.1: Red wine sensory quality rating distribution showing class concentration", fig_caption))

    story.append(PageBreak())

    # ================= CS-5 TASK 2 EDA & CORRELATION (ARABIC PAGE 12) =================
    story.append(Paragraph("Task 2: Exploratory Data Analysis & Feature Relationships", subsec_title))
    story.append(Paragraph(
        "Figure 3.2 presents boxplots of alcohol content grouped by quality grade, showing systematic upward progression from 9.60% (grade 5) to 12.50% (grade 8). "
        "Figure 3.3 illustrates the full 11-feature correlation heatmap. Strong inter-feature correlations include fixed acidity with density (r = +0.67) and fixed acidity with pH (r = -0.68).",
        body
    ))

    # Figure 3.3: Alcohol Boxplot
    if os.path.exists('figures/cs5_02_alcohol_by_quality_boxplot.png'):
        story.append(Image('figures/cs5_02_alcohol_by_quality_boxplot.png', width=11.5*cm, height=4.4*cm))
        story.append(Paragraph("Fig 3.2: Alcohol content grouped by quality score boxplot showing median trend", fig_caption))

    # Figure 3.4: Correlation Heatmap
    if os.path.exists('figures/cs5_03_correlation_heatmap.png'):
        story.append(Image('figures/cs5_03_correlation_heatmap.png', width=11.0*cm, height=4.6*cm))
        story.append(Paragraph("Fig 3.3: Correlation heatmap of physicochemical attributes and quality", fig_caption))

    story.append(Paragraph(
        "<b>Task 2d: Attribute Correlation Ranking with Quality:</b> "
        "Table 3.4 ranks all attributes by absolute Pearson correlation with quality. "
        "The top two predictors are: (1) <b>Alcohol (r = +0.4803)</b>, indicating full grape maturation and fermentation; "
        "and (2) <b>Volatile Acidity (r = -0.3952)</b>, indicating vinegar-like acetic acid spoilage.",
        body
    ))

    # Table 3.4: Correlation Ranking Table
    story.append(Paragraph("<b>Table 3.4: Feature ranking by absolute correlation with wine quality</b>", table_caption))
    tbl_rank_rows = [
        [Paragraph("<b>Rank</b>", tbl_header), Paragraph("<b>Attribute</b>", tbl_header),
         Paragraph("<b>Pearson r</b>", tbl_header), Paragraph("<b>Absolute r</b>", tbl_header), Paragraph("<b>Direction & Strength</b>", tbl_header)]
    ]
    for idx, row in cs5_rank_df.iterrows():
        dir_str = "Moderate Positive" if row['Pearson_r'] > 0.3 else ("Moderate Negative" if row['Pearson_r'] < -0.3 else ("Weak Positive" if row['Pearson_r'] > 0 else "Weak Negative"))
        tbl_rank_rows.append([
            Paragraph(str(idx + 1), tbl_cell),
            Paragraph(str(row['Attribute']), tbl_cell_left),
            Paragraph(f"{row['Pearson_r']:+.4f}", tbl_cell),
            Paragraph(f"{row['Abs_r']:.4f}", tbl_cell),
            Paragraph(dir_str, tbl_cell)
        ])
    t_rank = Table(tbl_rank_rows, colWidths=[1.5*cm, 4.5*cm, 2.5*cm, 2.5*cm, 4.5*cm])
    t_rank.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_rank)
    story.append(PageBreak())

    # ================= CS-5 MODEL EVALUATION & MULTICOLLINEARITY (ARABIC PAGE 13) =================
    story.append(Paragraph("3.5 Model Evaluation", sec_title))
    story.append(Paragraph(
        "A multiple linear regression model was trained on an 80:20 split (1,087 train, 272 test, random_state=42). "
        "Test-set performance yielded MAE = 0.5041, MSE = 0.4310, and RMSE = 0.6565 quality points. "
        "The test R2 of 0.3915 aligns closely with train R2 (0.3520), confirming that the model generalizes stably without overfitting (Table 3.5).",
        body
    ))

    # Table 3.5: Regression Metrics Table
    story.append(Paragraph("<b>Table 3.5: Multiple linear regression test set performance metrics</b>", table_caption))
    tbl_reg_rows = [
        [Paragraph("<b>Metric</b>", tbl_header), Paragraph("<b>Value</b>", tbl_header), Paragraph("<b>Interpretation</b>", tbl_header)]
    ]
    for _, row in cs5_reg_df.iterrows():
        tbl_reg_rows.append([
            Paragraph(str(row['Metric']), tbl_cell),
            Paragraph(str(row['Value']), tbl_cell),
            Paragraph(str(row['Meaning']), tbl_cell_left)
        ])
    t_reg = Table(tbl_reg_rows, colWidths=[2.8*cm, 2.2*cm, 10.5*cm])
    t_reg.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_reg)
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph(
        "<b>Multicollinearity Diagnosis (VIF Analysis):</b> "
        "Variance Inflation Factor (VIF) evaluation on training data identified moderate-to-high multicollinearity (Table 3.6): "
        "Fixed Acidity exhibited VIF = 7.95, and Density had VIF = 6.31 (both VIF > 5). "
        "This collinearity occurs because fixed acidity directly influences solution density and pH (r = -0.68).",
        body
    ))

    # Table 3.6: VIF Table
    story.append(Paragraph("<b>Table 3.6: Variance Inflation Factor (VIF) for training features</b>", table_caption))
    tbl_vif_rows = [
        [Paragraph("<b>Feature</b>", tbl_header), Paragraph("<b>VIF</b>", tbl_header), Paragraph("<b>Collinearity Status</b>", tbl_header)]
    ]
    for _, row in cs5_vif_df.iterrows():
        status = "High (VIF > 5)" if row['VIF'] > 5 else ("Moderate (3 - 5)" if row['VIF'] >= 3 else "Low (VIF < 3)")
        tbl_vif_rows.append([
            Paragraph(str(row['Feature']), tbl_cell_left),
            Paragraph(f"{row['VIF']:.2f}", tbl_cell),
            Paragraph(status, tbl_cell)
        ])
    t_vif = Table(tbl_vif_rows, colWidths=[5.5*cm, 3.5*cm, 6.5*cm])
    t_vif.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
    ]))
    story.append(t_vif)

    # Figure 3.5: Standardized Coefficients
    if os.path.exists('figures/cs5_04_standardized_coefficients.png'):
        story.append(Spacer(1, 0.1 * cm))
        story.append(Image('figures/cs5_04_standardized_coefficients.png', width=12.0*cm, height=4.8*cm))
        story.append(Paragraph("Fig 3.4: Standardized regression coefficients ranked by absolute impact |β|", fig_caption))

    story.append(PageBreak())

    # ================= CS-5 RESIDUAL DIAGNOSTICS & OLS SIGNIFICANCE (ARABIC PAGE 14) =================
    story.append(Paragraph(
        "Figure 3.5 displays predicted vs actual quality scores (with jitter) and residual diagnostic distributions. "
        "Predictions cluster narrowly between 5.0 and 6.5, showing clear shrinkage toward the training mean (5.63). "
        "The residual plot exhibits distinct downward-sloping parallel stripes (e = k - y_pred) corresponding to discrete integer ratings (k = 3, 4, 5, 6, 7, 8).",
        body
    ))

    # Figure 3.6: Predicted vs Actual & Residuals
    if os.path.exists('figures/cs5_05_predicted_vs_actual_residuals.png'):
        story.append(Image('figures/cs5_05_predicted_vs_actual_residuals.png', width=13.5*cm, height=5.0*cm))
        story.append(Paragraph("Fig 3.5: Predicted vs actual quality and residual diagnostics showing discrete score stripes", fig_caption))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("<b>Table 3.7: Ordinary Least Squares (OLS) regression parameter estimates and significance</b>", table_caption))
    if cs5_ols_df is not None:
        tbl_ols_rows = [
            [Paragraph("<b>Feature</b>", tbl_header), Paragraph("<b>Coef (β)</b>", tbl_header),
             Paragraph("<b>Std Err</b>", tbl_header), Paragraph("<b>t-stat</b>", tbl_header), Paragraph("<b>p-value</b>", tbl_header)]
        ]
        for _, row in cs5_ols_df.iterrows():
            tbl_ols_rows.append([
                Paragraph(str(row['Feature']), tbl_cell_left),
                Paragraph(f"{row['Coef']:+.4f}", tbl_cell),
                Paragraph(f"{row['Std_Err']:.4f}", tbl_cell),
                Paragraph(f"{row['t_stat']:+.2f}", tbl_cell),
                Paragraph(f"{row['p_value']:.4e}" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}", tbl_cell)
            ])
        t_ols = Table(tbl_ols_rows, colWidths=[4.5*cm, 2.8*cm, 2.6*cm, 2.8*cm, 2.8*cm])
        t_ols.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ]))
        story.append(t_ols)

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "Six attributes achieve statistical significance at alpha = 0.05: alcohol (p = 4.93e-20), volatile acidity (p = 3.47e-11), "
        "sulphates (p = 2.42e-09), chlorides (p = 5.76e-06), total sulfur dioxide (p = 4.77e-05), and pH (p = 0.0011). "
        "Conversely, fixed acidity (p = 0.380), citric acid (p = 0.589), residual sugar (p = 0.838), and density (p = 0.405) are not statistically significant after controlling for collinear covariates.",
        body
    ))
    story.append(PageBreak())

    # ================= CS-5 CV STABILITY & INTERPRETATION (ARABIC PAGE 15) =================
    story.append(Paragraph("<b>Table 3.8: 5-Fold cross-validation stability analysis across model metrics</b>", table_caption))
    if cs5_cv_stab_df is not None:
        tbl_cvs_rows = [
            [Paragraph("<b>Metric</b>", tbl_header), Paragraph("<b>Fold Mean</b>", tbl_header), Paragraph("<b>Fold Std</b>", tbl_header)]
        ]
        for _, row in cs5_cv_stab_df.iterrows():
            tbl_cvs_rows.append([
                Paragraph(str(row['Metric']), tbl_cell_left),
                Paragraph(f"{row['Fold_Mean']:.4f}", tbl_cell),
                Paragraph(f"{row['Fold_Std']:.4f}", tbl_cell)
            ])
        t_cvs = Table(tbl_cvs_rows, colWidths=[5.0*cm, 5.0*cm, 5.0*cm])
        t_cvs.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ]))
        story.append(t_cvs)

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("3.6 Interpretation and Business Recommendations", sec_title))
    story.append(Paragraph(
        "<b>Task 4a: Recommended Automated Laboratory Screening Measurements:</b><br/>"
        "Based on empirical effect sizes and statistical significance, wineries should implement automated testing on two core chemical markers:<br/>"
        "1. <b>Alcohol Content:</b> Strongest positive correlation (r = +0.4803), largest positive standardized coefficient (beta = +0.3340), and p < 0.0001. "
        "Higher alcohol signals proper fruit maturation and complete sugar fermentation.<br/>"
        "2. <b>Volatile Acidity:</b> Second strongest correlation (r = -0.3952), largest negative standardized coefficient (beta = -0.1785), and p < 0.0001. "
        "Elevated acetic acid indicates bacterial contamination and spoilage.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "<b>Task 4b: Can the Model Replace Human Sensory Panels?</b><br/>"
        "<b>No.</b> With an R2 of 0.3915 (~39% variance explained) and an RMSE of 0.6565 quality points, the model frequently misjudges wines by an entire rating tier. "
        "Linear regression on chemical attributes cannot detect complex sensory attributes such as bouquet, aroma balance, finish, and mouthfeel. "
        "The model should serve strictly as an automated, low-cost vat pre-screening tool to detect flawed batches prior to tasting.",
        body
    ))
    story.append(PageBreak())

    # ================= CS-5 LIMITATIONS & CONCLUSION (ARABIC PAGE 16) =================
    story.append(Paragraph("3.7 Conclusion and Limitations", sec_title))
    story.append(Paragraph(
        "<b>Task 4c: Real-World Limitations:</b><br/>"
        "1. <b>Regional & Varietal Specificity:</b> The data is restricted exclusively to red Vinho Verde wines from northern Portugal; chemical thresholds do not generalize to other varietals.<br/>"
        "2. <b>Subjectivity and Discreteness of Target:</b> Quality is an ordinal median score from human experts, embodying sensory noise that linear equations cannot capture.<br/>"
        "3. <b>Class Imbalance:</b> Over 81.8% of wines reside in scores 5 and 6, leaving scarce training examples for extreme quality grades.<br/>"
        "4. <b>Unmeasured Volatiles:</b> Key sensory factors such as tannin structure, polyphenols, and oak lactones were not quantified in laboratory assays.",
        body
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "Multiple linear regression achieved moderate predictive utility (RMSE = 0.66, R2 = 0.39), successfully identifying alcohol and volatile acidity "
        "as the primary chemical drivers of wine quality ratings. "
        "While valuable as a rapid vat-screening filter, human sensory evaluation remains indispensable for final commercial grading.",
        body
    ))
    story.append(PageBreak())

    # ================= CHAPTER 4: CONCLUSION AND REFERENCES (ARABIC PAGE 17) =================
    story.append(Paragraph("4. CONCLUSION and References", ch_title))
    story.append(Paragraph("4.1 Conclusion", sec_title))
    story.append(Paragraph(
        "Both case studies illustrate that thorough exploratory analysis, principled preprocessing, and rigorous diagnostic evaluation "
        "are essential for reliable data science solutions. "
        "In Case Study 2, K-Means clustering (K = 4) and Hierarchical Ward linkage successfully resolved frequent flyer accounts into four distinct behavioral tiers: "
        "Active Frequent Flyers (27.7%), Credit Card Loyalty Earners (45.0%), Inactive Members (27.2%), and Corporate Outliers (<0.1%), "
        "enabling tailored marketing campaigns that allocate rewards efficiently. "
        "In Case Study 5, multiple linear regression achieved stable out-of-sample prediction (RMSE = 0.66, R2 = 0.39), identifying alcohol (+0.3340) "
        "and volatile acidity (-0.1785) as primary chemical determinants. "
        "The analysis proved that chemical regression operates effectively as an automated pre-screening filter to catch spoiled vats, "
        "while sensory tasting panels remain essential for holistic quality appraisal.",
        body
    ))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("4.2 References", sec_title))
    story.append(Paragraph(
        "[1] G. Shmueli and P. C. Bruce, <i>Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python</i>, John Wiley & Sons, 2016. "
        "Dataset: East-West Airlines Frequent Flyer Data.<br/>"
        "[2] P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, “Modeling wine preferences by data mining from physicochemical properties,” "
        "<i>Decision Support Systems</i>, vol. 47, no. 4, pp. 547-553, 2009. Dataset: Wine Quality (Red), UCI Machine Learning Repository.<br/>"
        "[3] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.<br/>"
        "[4] S. Seabold and J. Perktold, “Statsmodels: Econometric and statistical modeling with Python,” <i>Proceedings of the 9th Python in Science Conference</i>, 2010.<br/>"
        "[5] Python libraries: pandas, NumPy, Matplotlib, seaborn, SciPy, and ReportLab.<br/>"
        f"[6] Project GitHub repository: {github_url}",
        body_left
    ))
    return story

def build_pdf_with_page_inspection(student_name="Katthi Goutham Reddy", roll_no="160124733179", section="CSE-3", 
                                   faculty="Dr. S. Durga Devi", github_url="https://github.com/katthi-goutham-reddy/FDS-case-study"):
    
    os.makedirs('report', exist_ok=True)
    pdf_filename = f"FDS_Case_Studies_2_and_5_{roll_no}_CBIT_format.pdf"
    pdf_path = os.path.join('report', pdf_filename)
    root_pdf_path = pdf_filename  # Also save directly in root

    # Pass 1: Build document with initial estimates
    doc1 = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=2.0*cm,
        bottomMargin=2.0*cm
    )
    story1 = create_report_story(student_name, roll_no, section, faculty, github_url)
    doc1.build(story1, canvasmaker=CBITCanvas)
    print("Pass 1 build complete. Inspecting page locations...")

    # Inspect document to find exact page numbers
    doc_fitz = fitz.open(pdf_path)
    total_pages = len(doc_fitz)
    print(f"Total pages generated: {total_pages}")

    toc_pages = {}
    lof_pages = {}

    for p_idx in range(4, total_pages):
        page_text = doc_fitz[p_idx].get_text()
        content_page_num = p_idx - 3  # Arabic numeral on the page

        # TOC targets
        if "1. INTRODUCTION" in page_text and 'intro' not in toc_pages:
            toc_pages['intro'] = content_page_num
        if "1.1 Problem Definition" in page_text and 'p1_1' not in toc_pages:
            toc_pages['p1_1'] = content_page_num
        if "1.2 Concepts used in Methodology" in page_text and 'p1_2' not in toc_pages:
            toc_pages['p1_2'] = content_page_num
        if "1.3 Outline the results" in page_text and 'p1_3' not in toc_pages:
            toc_pages['p1_3'] = content_page_num
        if "2. CASE STUDY 2: CLUSTERING ANALYSIS" in page_text and 'cs2' not in toc_pages:
            toc_pages['cs2'] = content_page_num
        if "2.1 Problem Statement" in page_text and 'p2_1' not in toc_pages:
            toc_pages['p2_1'] = content_page_num
        if "2.2 Dataset Description" in page_text and 'p2_2' not in toc_pages:
            toc_pages['p2_2'] = content_page_num
        if "2.3 Methodology" in page_text and 'p2_3' not in toc_pages:
            toc_pages['p2_3'] = content_page_num
        if "2.4 Statistical Findings" in page_text and 'p2_4' not in toc_pages:
            toc_pages['p2_4'] = content_page_num
        if "2.5 Model Evaluation" in page_text and 'p2_5' not in toc_pages:
            toc_pages['p2_5'] = content_page_num
        if "2.6 Interpretation and Business" in page_text and 'p2_6' not in toc_pages:
            toc_pages['p2_6'] = content_page_num
        if "2.7 Conclusion and Limitations" in page_text and 'p2_7' not in toc_pages:
            toc_pages['p2_7'] = content_page_num
        if "3. CASE STUDY 5: WINE QUALITY" in page_text and 'cs5' not in toc_pages:
            toc_pages['cs5'] = content_page_num
        if "3.1 Problem Statement" in page_text and 'p3_1' not in toc_pages:
            toc_pages['p3_1'] = content_page_num
        if "3.2 Dataset Description" in page_text and 'p3_2' not in toc_pages:
            toc_pages['p3_2'] = content_page_num
        if "3.3 Methodology" in page_text and 'p3_3' not in toc_pages:
            toc_pages['p3_3'] = content_page_num
        if "3.4 Statistical Findings" in page_text and 'p3_4' not in toc_pages:
            toc_pages['p3_4'] = content_page_num
        if "3.5 Model Evaluation" in page_text and 'p3_5' not in toc_pages:
            toc_pages['p3_5'] = content_page_num
        if "3.6 Interpretation and Business" in page_text and 'p3_6' not in toc_pages:
            toc_pages['p3_6'] = content_page_num
        if "3.7 Conclusion and Limitations" in page_text and 'p3_7' not in toc_pages:
            toc_pages['p3_7'] = content_page_num
        if "4. CONCLUSION and References" in page_text and 'conc' not in toc_pages:
            toc_pages['conc'] = content_page_num
        if "4.1 Conclusion" in page_text and 'p4_1' not in toc_pages:
            toc_pages['p4_1'] = content_page_num
        if "4.2 References" in page_text and 'p4_2' not in toc_pages:
            toc_pages['p4_2'] = content_page_num

        # LOF targets
        for f_id in ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7',
                     '3.1', '3.2', '3.3', '3.4', '3.5']:
            key = f"fig{f_id.replace('.', '_')}"
            if f"Fig {f_id}:" in page_text and key not in lof_pages:
                lof_pages[key] = content_page_num

    print("Detected TOC page mapping:", toc_pages)
    print("Detected LOF page mapping:", lof_pages)

    # Pass 2: Re-build with exact page numbers
    doc2 = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=2.0*cm,
        bottomMargin=2.0*cm
    )
    story2 = create_report_story(student_name, roll_no, section, faculty, github_url, toc_pages, lof_pages)
    doc2.build(story2, canvasmaker=CBITCanvas)
    print(f"Pass 2 build complete: {pdf_path}")

    # Also copy to root for instant access
    import shutil
    shutil.copyfile(pdf_path, root_pdf_path)
    print(f"Copied to root: {root_pdf_path}")

    # Generate preview PNG images using PyMuPDF
    doc_final = fitz.open(pdf_path)
    preview_dir = os.path.join('report', 'cbit_preview')
    os.makedirs(preview_dir, exist_ok=True)
    for i, page in enumerate(doc_final):
        pix = page.get_pixmap(dpi=150)
        pix.save(os.path.join(preview_dir, f'page_{i+1}.png'))
    print(f"Exported {len(doc_final)} preview pages to {preview_dir}")

    return pdf_path

if __name__ == '__main__':
    s_name = sys.argv[1] if len(sys.argv) > 1 else "Katthi Goutham Reddy"
    r_no = sys.argv[2] if len(sys.argv) > 2 else "160124733179"
    sec = sys.argv[3] if len(sys.argv) > 3 else "CSE-3"
    fac = sys.argv[4] if len(sys.argv) > 4 else "Dr. S. Durga Devi"
    gh = sys.argv[5] if len(sys.argv) > 5 else "https://github.com/katthi-goutham-reddy/FDS-case-study"
    build_pdf_with_page_inspection(s_name, r_no, sec, fac, gh)
