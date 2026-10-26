"""
Unified PDF Report Generator for Fundamentals of Data Science Case Studies 2 & 5
Generates a clean, comprehensive, college-student style laboratory report using ReportLab.
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Skip headers and footers on Cover Page (Page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(2 * cm, 28.2 * cm, "Fundamentals of Data Science (FDS) | Case Studies 2 & 5 Report")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(2 * cm, 28.0 * cm, 19 * cm, 28.0 * cm)

            # Footer
            self.drawString(2 * cm, 1.2 * cm, "Chaitanya Bharathi Institute of Technology (CBIT) | B.E. CSE")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(19 * cm, 1.2 * cm, page_text)
            self.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)

        self.restoreState()

def build_unified_pdf(student_name="<NAME>", roll_no="<ROLL NO>", section="<SECTION>", 
                      faculty="<FACULTY NAME>", github_url="<GITHUB URL>"):
    
    # Load Results
    cs2_comp_df = pd.read_csv('results/cs2_clustering_comparison.csv')
    cs2_prof_df = pd.read_csv('results/cs2_cluster_profiles.csv')
    
    cs5_stats_df = pd.read_csv('results/cs5_task1_stats.csv')
    cs5_cv_df = pd.read_csv('results/cs5_task1_cv.csv')
    cs5_rank_df = pd.read_csv('results/cs5_task2_feature_correlation_rank.csv')
    cs5_reg_df = pd.read_csv('results/cs5_task3_regression_metrics.csv')
    cs5_vif_df = pd.read_csv('results/cs5_task3_vif.csv')
    cs5_std_df = pd.read_csv('results/cs5_task3_standardized_coefs.csv')

    os.makedirs('report', exist_ok=True)
    clean_roll = roll_no.replace('<', '').replace('>', '').strip()
    pdf_path = f"report/FDS_Case_Studies_{clean_roll if clean_roll and clean_roll != 'ROLL NO' else 'Report'}.pdf"

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
        fontSize=20,
        leading=26,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1
    )

    h1 = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
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
        textColor=colors.HexColor("#2563eb"),
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

    # ================= 1. COVER PAGE =================
    story.append(Spacer(1, 3.0 * cm))
    story.append(Paragraph("Chaitanya Bharathi Institute of Technology (CBIT)", ParagraphStyle('College', fontName='Helvetica-Bold', fontSize=14, leading=18, alignment=1, textColor=colors.HexColor("#333333"))))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Department of Computer Science and Engineering", ParagraphStyle('Dept', fontName='Helvetica', fontSize=11, leading=15, alignment=1, textColor=colors.HexColor("#666666"))))
    story.append(Spacer(1, 1.2 * cm))

    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#1e3a8a"), spaceAfter=18))
    story.append(Paragraph("FUNDAMENTALS OF DATA SCIENCE", title_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Comprehensive Case Study Report (CS-2 & CS-5)", ParagraphStyle('Sub', fontName='Helvetica-Bold', fontSize=12, leading=16, alignment=1, textColor=colors.HexColor("#2563eb"))))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Part I: Customer Segmentation using Clustering (East-West Airlines)<br/>Part II: Wine Quality Prediction via Multiple Linear Regression", ParagraphStyle('Sub2', fontName='Helvetica-Oblique', fontSize=10, leading=14, alignment=1, textColor=colors.HexColor("#444444"))))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#1e3a8a"), spaceBefore=18, spaceAfter=22))

    story.append(Spacer(1, 1.2 * cm))

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
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5.5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ================= 2. PART I: CASE STUDY 2 (PAGE 2) =================
    story.append(Paragraph("PART I: Case Study 2 - Customer Segmentation using Clustering", h1))
    story.append(Paragraph(
        "<b>1. Problem Statement and Objectives</b><br/>"
        "East-West Airlines collected operational flight, mileage accrual, credit card partnership transactions, and enrollment tenure data "
        "for 3,999 frequent flyer accounts. The airline seeks to discover natural customer segments to tailor marketing campaigns, "
        "tier incentives, and loyalty redemption options rather than treating all members uniformly.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>2. Dataset Preprocessing & Outlier Handling</b><br/>"
        "The dataset contains 3,999 records across 12 features with 0 missing values. Feature <code>ID#</code> was dropped as an arbitrary identifier. "
        "The target loyalty indicator <code>Award?</code> was withheld from clustering and reserved strictly for post-clustering profiling. "
        "Continuous mileage features exhibited severe right-skewness (cc3_miles: +17.20, Balance: +5.00). Applying <code>np.log1p</code> normalized distributions. "
        "Outliers were removed using |z| > 3 on log-transformed variables, removing 376 extreme records (9.40%) and retaining 3,623 clean records without dropping legitimate high-tier flyers. "
        "Features were subsequently standardized using StandardScaler.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Figure 1: Raw vs Log Histograms
    if os.path.exists('figures/cs2_01_raw_vs_log_histograms.png'):
        story.append(Image('figures/cs2_01_raw_vs_log_histograms.png', width=13.5*cm, height=6.0*cm))
        story.append(Paragraph("Figure 1: Distribution of key mileage features before and after log1p transformation", caption_style))

    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>3. Exploratory Data Analysis & Structure</b><br/>"
        "Correlation analysis revealed that credit card mileage (cc1_miles) correlates strongly with bonus miles (r = 0.83) and bonus transactions (r = 0.60), "
        "distinguishing credit card spenders from frequent flyers. Initial 2D PCA showed a dense continuous central customer core with extended arms representing elite flyers.",
        normal
    ))
    story.append(PageBreak())

    # ================= 3. CS-2 CLUSTERING MODELS & RESULTS (PAGE 3) =================
    story.append(Paragraph("4. Clustering Algorithms and Model Evaluation", h1))
    story.append(Paragraph(
        "Three distinct clustering algorithms were systematically evaluated on the standardized feature matrix:<br/>"
        "1. <b>K-Means:</b> Evaluated across K = 1 to 10. Both the inertia elbow curve and silhouette score profile indicated an optimal partition at K = 4 (Inertia = 12,568; Silhouette = 0.3199).<br/>"
        "2. <b>Hierarchical (Agglomerative):</b> Evaluated across Ward, Complete, Average, and Single linkage. Ward linkage produced the most cohesive tree with the highest silhouette score (0.3317 at K = 4), while Single linkage suffered from chaining (silhouette = 0.2641).<br/>"
        "3. <b>DBSCAN:</b> Evaluated using 5-nearest neighbor knee analysis. An eps of 0.879 with min_samples = 10 separated 3 dense customer cores while setting aside 220 noise points (6.07%).",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Figure 2: Elbow & Silhouette Plot
    if os.path.exists('figures/cs2_06_kmeans_elbow_silhouette.png'):
        story.append(Image('figures/cs2_06_kmeans_elbow_silhouette.png', width=13.5*cm, height=5.2*cm))
        story.append(Paragraph("Figure 2: K-Means optimal K selection via inertia elbow and silhouette score analysis", caption_style))

    story.append(Spacer(1, 0.15*cm))

    # Table 1: Clustering Comparison Table
    story.append(Paragraph("<b>Table 1: Clustering Algorithm Evaluation Comparison</b>", h2))
    table_rows = [
        [Paragraph("<b>Algorithm</b>", table_header), Paragraph("<b>Clusters</b>", table_header),
         Paragraph("<b>Silhouette</b>", table_header), Paragraph("<b>Davies-Bouldin</b>", table_header),
         Paragraph("<b>Calinski-Harabasz</b>", table_header)]
    ]
    for _, row in cs2_comp_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Algorithm']), table_cell),
            Paragraph(str(row['Clusters']), table_cell),
            Paragraph(f"{row['Silhouette']:.4f}", table_cell),
            Paragraph(f"{row['Davies-Bouldin']:.4f}", table_cell),
            Paragraph(f"{row['Calinski-Harabasz']:.2f}", table_cell)
        ])
    t1 = Table(table_rows, colWidths=[4.2*cm, 2.0*cm, 2.8*cm, 3.2*cm, 3.6*cm])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.15*cm))

    # Figure 3: PCA Cluster Scatter Plots
    if os.path.exists('figures/cs2_09_cluster_pca_scatter.png'):
        story.append(Image('figures/cs2_09_cluster_pca_scatter.png', width=14.5*cm, height=4.8*cm))
        story.append(Paragraph("Figure 3: 2D PCA cluster visualizations comparing K-Means, Hierarchical (Ward), and DBSCAN", caption_style))

    story.append(PageBreak())

    # ================= 4. CS-2 PROFILES & BUSINESS DECISIONS (PAGE 4) =================
    story.append(Paragraph("5. Customer Segment Personas & Marketing Recommendations", h1))
    story.append(Paragraph(
        "K-Means clustering at K = 4 revealed four distinct customer personas based on unscaled feature averages and award redemption rates:",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Figure 4: Cluster Profile Heatmap
    if os.path.exists('figures/cs2_10_cluster_heatmap.png'):
        story.append(Image('figures/cs2_10_cluster_heatmap.png', width=13.5*cm, height=5.2*cm))
        story.append(Paragraph("Figure 4: Standardized feature profile heatmap across K-Means customer segments", caption_style))

    story.append(Paragraph(
        "<b>Customer Personas Identified:</b><br/>"
        "1. <b>Cluster 0: Active Frequent Flyers (27.7% of accounts):</b> Characterized by high flight activity (1,076 flight miles/yr, 3.2 flight transactions) "
        "and a high Award flight redemption rate of 54.2%. <i>Strategy:</i> Retain tier status with complimentary lounge access, priority check-in, and seat upgrade certificates.<br/>"
        "2. <b>Cluster 1: Credit Card Loyalty Earners (45.0% of accounts):</b> Characterized by high credit card spend (20,900 bonus miles, 15.5 transactions) "
        "but near-zero actual flight activity. <i>Strategy:</i> Offer targeted award seat promotions (e.g. 20% mileage discount on flights) to convert card spenders into active airline travelers.<br/>"
        "3. <b>Cluster 2: Inactive / Low Engagement Members (27.2% of accounts):</b> Characterized by low balances (22,418 miles), minimal transactions, and low award rate (11.8%). "
        "<i>Strategy:</i> Re-engagement campaigns, entry-level mileage incentives, and account expiration reminders.<br/>"
        "4. <b>Cluster 3: Elite Outlier Accounts (<0.1%):</b> High-volume qualifying accounts requiring dedicated corporate concierge support.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>Practical Limitations of Clustering:</b><br/>"
        "1. Cluster count K = 4 involves mathematical trade-offs between inertia reduction and segment interpretability.<br/>"
        "2. DBSCAN struggled with non-uniform density, grouping 94% of customers into a single broad cluster while labeling 6% as noise.<br/>"
        "3. PCA 2D projections retain ~40% of variance, meaning 60% of high-dimensional geometric separation is compressed.",
        normal
    ))
    story.append(PageBreak())

    # ================= 5. PART II: CASE STUDY 5 (PAGE 5) =================
    story.append(Paragraph("PART II: Case Study 5 - Wine Quality Prediction via Multiple Regression", h1))
    story.append(Paragraph(
        "<b>1. Problem Statement and Objectives</b><br/>"
        "Wine quality assessment traditionally relies on expert sensory tasting panels, which is expensive, slow, and prone to taster palate fatigue. "
        "The objective of this case study is to determine whether laboratory-measured physicochemical attributes can predict wine quality ratings "
        "using statistical hypothesis testing and Multiple Linear Regression.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>2. Dataset Description & Hygiene</b><br/>"
        "The red wine variant of the Portuguese Vinho Verde dataset (Cortez et al., 2009) contains 1,599 observations across 11 physicochemical inputs and 1 quality target. "
        "Data hygiene checks identified 0 missing values and 240 duplicate rows (15.01%). Duplicate rows were removed prior to model training to prevent data leakage "
        "across train/test splits, yielding 1,359 clean, unique wine samples.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Table 2: Task 1a Descriptive Stats
    story.append(Paragraph("<b>Table 2: Descriptive Statistics and Skewness (Task 1a)</b>", h2))
    table_rows = [
        [Paragraph("<b>Attribute</b>", table_header), Paragraph("<b>Mean</b>", table_header),
         Paragraph("<b>Median</b>", table_header), Paragraph("<b>Mode</b>", table_header),
         Paragraph("<b>Std Dev</b>", table_header), Paragraph("<b>Skewness</b>", table_header)]
    ]
    for _, row in cs5_stats_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Attribute']), table_cell),
            Paragraph(f"{row['Mean']:.4f}", table_cell),
            Paragraph(f"{row['Median']:.2f}", table_cell),
            Paragraph(f"{row['Mode']:.2f}", table_cell),
            Paragraph(f"{row['Std_Dev']:.4f}", table_cell),
            Paragraph(f"{row['Skewness']:.4f}", table_cell)
        ])
    t2 = Table(table_rows, colWidths=[3.6*cm, 2.4*cm, 2.2*cm, 2.2*cm, 2.6*cm, 2.6*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        f"<b>Task 1b: Relative Dispersion via Coefficient of Variation (CV)</b><br/>"
        f"Residual Sugar exhibited the highest relative variability (CV = {cs5_cv_df.loc[cs5_cv_df['Attribute']=='residual sugar', 'CV(%)'].values[0]:.2f}%), "
        f"compared to Volatile Acidity ({cs5_cv_df.loc[cs5_cv_df['Attribute']=='volatile acidity', 'CV(%)'].values[0]:.2f}%) and Alcohol ({cs5_cv_df.loc[cs5_cv_df['Attribute']=='alcohol', 'CV(%)'].values[0]:.2f}%). "
        f"CV is the mathematically appropriate comparative measure because standard deviation depends on measurement scale (% vol vs g/dm3), "
        f"whereas CV normalizes dispersion by dividing standard deviation by the mean.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "<b>Task 1c: Pearson Correlation Hypothesis Test (Alcohol vs Quality)</b><br/>"
        "A formal two-tailed test was performed (H0: r = 0 vs H1: r != 0, alpha = 0.05). "
        "The Pearson correlation was r = +0.4803 with p-value = 2.28e-79. "
        "Since p < 0.05, we reject H0 and conclude that a statistically significant, moderate positive linear relationship exists.",
        normal
    ))
    story.append(PageBreak())

    # ================= 6. CS-5 EDA & REGRESSION (PAGE 6) =================
    story.append(Paragraph("3. Exploratory Data Analysis & Feature Relationships", h1))
    
    # Figure 5: Quality Distribution
    if os.path.exists('figures/cs5_01_quality_distribution.png'):
        story.append(Image('figures/cs5_01_quality_distribution.png', width=13.0*cm, height=5.2*cm))
        story.append(Paragraph("Figure 5: Wine sensory quality distribution showing heavy imbalance around scores 5 and 6 (81.8% of wines)", caption_style))

    story.append(Spacer(1, 0.15*cm))

    # Figure 6: Alcohol Boxplot
    if os.path.exists('figures/cs5_02_alcohol_by_quality_boxplot.png'):
        story.append(Image('figures/cs5_02_alcohol_by_quality_boxplot.png', width=13.0*cm, height=5.2*cm))
        story.append(Paragraph("Figure 6: Alcohol distribution across quality tiers showing median upward trend from 9.6% to 12.5%", caption_style))

    story.append(Paragraph(
        "<b>Task 2d: Correlation Ranking with Quality</b><br/>"
        "Ranking all 11 attributes by absolute Pearson correlation identified the top two predictors: "
        "(1) <b>Alcohol (r = +0.4803)</b>: Positive association reflecting ripe fruit and proper fermentation; "
        "(2) <b>Volatile Acidity (r = -0.3952)</b>: Negative association reflecting unpleasant vinegar spoilage caused by acetic acid bacteria. "
        "Noticeable collinear pairs include fixed acidity with density (r = +0.67) and fixed acidity with pH (r = -0.68).",
        normal
    ))
    story.append(PageBreak())

    # ================= 7. CS-5 MODEL EVALUATION & DIAGNOSTICS (PAGE 7) =================
    story.append(Paragraph("4. Multiple Linear Regression & Model Diagnostics", h1))
    story.append(Paragraph(
        "A multiple linear regression model was fitted on an 80:20 train-test split (1,087 train, 272 test, random_state=42). "
        "StandardScaler was fitted strictly on the training set to prevent data leakage.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Table 3: Regression Metrics Table
    story.append(Paragraph("<b>Table 3: Multiple Linear Regression Test-Set Performance Metrics</b>", h2))
    table_rows = [
        [Paragraph("<b>Metric</b>", table_header), Paragraph("<b>Value</b>", table_header), Paragraph("<b>Interpretation</b>", table_header)]
    ]
    for _, row in cs5_reg_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Metric']), table_cell),
            Paragraph(str(row['Value']), table_cell),
            Paragraph(str(row['Meaning']), table_cell)
        ])
    t3 = Table(table_rows, colWidths=[3.2*cm, 2.5*cm, 10*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        "<b>Multicollinearity Diagnosis (VIF Analysis):</b><br/>"
        "Variance Inflation Factor (VIF) evaluation revealed moderate-to-high multicollinearity for Fixed Acidity (VIF = 7.95) and Density (VIF = 6.31). "
        "While predictions remain unbiased, individual coefficients for these collinear features must be interpreted with caution. "
        "Statsmodels OLS confirmed that volatile acidity, chlorides, total sulfur dioxide, sulphates, and alcohol are statistically significant (p < 0.05).",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Figure 7: Standardized Coefficients
    if os.path.exists('figures/cs5_04_standardized_coefficients.png'):
        story.append(Image('figures/cs5_04_standardized_coefficients.png', width=13.0*cm, height=5.2*cm))
        story.append(Paragraph("Figure 7: Standardized regression coefficients ranked by absolute impact |β| with direction coding", caption_style))

    story.append(PageBreak())

    # ================= 8. CS-5 DIAGNOSTICS & DECISIONS (PAGE 8) =================
    story.append(Paragraph("5. Model Diagnostics & Business Decision-Making", h1))

    # Figure 8: Predicted vs Actual & Residual Diagnostics
    if os.path.exists('figures/cs5_05_predicted_vs_actual_residuals.png'):
        story.append(Image('figures/cs5_05_predicted_vs_actual_residuals.png', width=14.5*cm, height=4.8*cm))
        story.append(Paragraph("Figure 8: Predicted vs actual quality (left) and residual plot showing discrete score tier stripes (right)", caption_style))

    story.append(Paragraph(
        "<b>Diagnostic Takeaways:</b><br/>"
        "Continuous linear regression compresses predictions into a narrow band between 5.0 and 6.5, shrinking extreme scores toward the training mean (5.63). "
        "The residual plot exhibits distinct downward-sloping parallel stripes (e = k - y_pred) corresponding to discrete integer ratings.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(
        "<b>Task 4: Business Decisions & Recommendations:</b><br/>"
        "1. <b>Automated Lab Screening Measurements:</b> The winery should implement rapid automated testing on (a) <b>Alcohol Content</b> (strongest positive impact, beta = +0.3340, p < 0.0001) "
        "and (b) <b>Volatile Acidity</b> (strongest negative spoilage impact, beta = -0.1785, p < 0.0001).<br/>"
        "2. <b>Can the Model Replace Sensory Panels?</b> <b>No.</b> With R2 = 0.3915 (~39% variance explained) and RMSE = 0.6565 quality points, "
        "prediction errors frequently cross an entire quality rating tier. The model should serve strictly as a pre-screening filter to catch flawed vats before submitting samples to human tasting panels.<br/>"
        "3. <b>Real-World Limitations:</b> Restriction to red Vinho Verde wines, ordinal nature of subjective sensory ratings, and severe class imbalance (81.8% in scores 5-6).",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))

    # ================= 9. CONCLUSION & REFERENCES =================
    story.append(Paragraph("6. Conclusion & References", h1))
    story.append(Paragraph(
        "This unified practical investigation applied unsupervised and supervised data science methodologies to real-world datasets. "
        "In Case Study 2, K-Means clustering (K = 4) and Hierarchical Ward linkage successfully segmented airline customers into actionable operational tiers. "
        "In Case Study 5, multiple linear regression achieved stable out-of-sample prediction (RMSE = 0.66, R2 = 0.39), identifying alcohol and volatile acidity as the primary chemical determinants of wine quality.",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<b>References:</b><br/>"
        "1. Shmueli, G., and Bruce, P. C. (2016). <i>Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python</i>. John Wiley & Sons.<br/>"
        "2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. <i>Decision Support Systems</i>, Elsevier, 47(4):547-553.<br/>"
        "3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.<br/>"
        "4. Seabold, S., and Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. <i>Proceedings of the 9th Python in Science Conference</i>.<br/>"
        f"5. Project Repository: {github_url}",
        normal
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated unified report: {pdf_path}")

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
    build_unified_pdf(s_name, r_no, sec, fac, gh)
