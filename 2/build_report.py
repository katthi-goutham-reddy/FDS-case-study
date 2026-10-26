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
        self.drawString(2 * cm, A4[1] - 1.5 * cm, "Fundamentals of Data Science | Case Study 2: Clustering Analysis")
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
    
    comp_df = pd.read_csv('results/clustering_comparison.csv')
    profile_df = pd.read_csv('results/kmeans_cluster_profile.csv')
    skew_df = pd.read_csv('results/skewness_table.csv')

    os.makedirs('report', exist_ok=True)
    clean_roll = roll_no.replace('<', '').replace('>', '').strip()
    pdf_path = f"report/CS2_Clustering_Analysis_{clean_roll if clean_roll and clean_roll != 'ROLL NO' else 'Report'}.pdf"

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
        textColor=colors.HexColor("#1b365d"),
        alignment=1
    )

    h1 = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1b365d"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2 = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2b4c7e"),
        spaceBefore=8,
        spaceAfter=4,
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
    
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#1b365d"), spaceAfter=20))
    story.append(Paragraph("CASE STUDY 2: CLUSTERING ANALYSIS", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Customer Segmentation on East-West Airlines Data Using K-Means, Hierarchical, and DBSCAN Clustering", ParagraphStyle('Sub', fontName='Helvetica-Oblique', fontSize=11, leading=15, alignment=1, textColor=colors.HexColor("#444444"))))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor("#1b365d"), spaceBefore=20, spaceAfter=25))

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
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
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
        "East-West Airlines maintains an extensive frequent flyer loyalty programme. "
        "Understanding different customer segments based on flight frequency, mileage accumulation, "
        "and credit card partner usage is essential to optimize marketing spend and design targeted retention offers. "
        "The objective of this case study is to develop an end-to-end unsupervised learning pipeline to identify distinct "
        "customer groups from behavioural data.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Key objectives include: (1) cleaning and transforming highly skewed continuous features, (2) detecting and handling outliers "
        "without discarding genuine high-tier accounts, (3) evaluating three fundamentally distinct clustering paradigms (partitioning via K-Means, "
        "agglomerative hierarchical clustering with multiple linkages, and density-based clustering via DBSCAN), (4) comparing algorithm "
        "performance using internal validity indices (Silhouette, Davies-Bouldin, and Calinski-Harabasz), and (5) formulating actionable "
        "business strategies tailored to each identified customer persona.",
        normal
    ))

    # ================= 3. DATASET DESCRIPTION =================
    story.append(Paragraph("2. Dataset Description", h1))
    story.append(Paragraph(
        "The dataset originates from Shmueli and Bruce (2016) and contains frequent flyer records for 3,999 customers across 12 features. "
        "The variables capture account age, balance, tier qualifying miles, mileage earned through partner credit cards (cc1, cc2, cc3), "
        "non-flight bonus transactions, flight transactions in the past 12 months, and award redemption history.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Pre-modelling inspection confirmed 0 missing values and 0 duplicate records. However, substantial data quality challenges were present: "
        "the mileage and transaction attributes exhibit severe positive skewness (e.g. cc3_miles skewness = 17.20, Balance skewness = 5.00). "
        "Standard IQR outlier filtering on raw features would have discarded 569 records (14.2%) in Flight_miles_12mo and 266 (6.7%) in Balance, "
        "erroneously eliminating the airline's most valuable elite flyers. Customer ID was dropped as a non-informative identifier, "
        "and the Award? binary indicator was excluded from cluster distance matrices and retained strictly for post-clustering profiling.",
        normal
    ))

    # ================= 4. METHODOLOGY =================
    story.append(Paragraph("3. Methodology", h1))
    story.append(Paragraph(
        "To establish a robust clustering framework, features with skewness > 1.0 were transformed using np.log1p (log(x + 1)), "
        "which compresses extreme ranges while preserving zeros. Outlier removal was subsequently applied using a standard Z-score threshold "
        "of |z| > 3 on the log-transformed space, removing 376 extreme multivariate outliers (9.40%) and retaining 3,623 records. "
        "All 10 feature columns were then normalized using StandardScaler (zero mean, unit variance).",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/00_methodology_flowchart.png'):
        img = Image('figures/00_methodology_flowchart.png', width=14*cm, height=7*cm)
        story.append(img)
        story.append(Paragraph("Figure 1: End-to-end customer segmentation methodology workflow", caption_style))

    story.append(Paragraph(
        "Model building evaluated three paradigms: K-Means (tested for K=1 to 10 with 10 random restarts), Agglomerative Hierarchical clustering "
        "(evaluating Ward, Complete, Average, and Single linkages), and DBSCAN (parameterized via 5-nearest neighbour k-distance knee analysis "
        "and a grid search across eps and min_samples).",
        normal
    ))
    story.append(PageBreak())

    # ================= 5. STATISTICAL FINDINGS =================
    story.append(Paragraph("4. Statistical Findings & Exploratory Data Analysis", h1))
    story.append(Paragraph(
        "Skewness analysis identified 8 features with severe right-skewness > 1.0: cc3_miles (17.20), cc2_miles (8.73), Qual_miles (7.52), "
        "Flight_miles_12mo (7.45), Flight_trans_12 (5.49), Balance (5.00), Bonus_miles (2.84), and Bonus_trans (1.16). "
        "Applying np.log1p transformed these distributions toward near-normal symmetry, stabilizing Euclidean distance calculations.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/01_raw_vs_log_histograms.png'):
        img = Image('figures/01_raw_vs_log_histograms.png', width=13.5*cm, height=8*cm)
        story.append(img)
        story.append(Paragraph("Figure 2: Distribution transformation: Raw vs log1p for key continuous attributes showing skewness reduction", caption_style))

    story.append(Paragraph(
        "Correlation analysis indicated strong collinearity between credit card miles and bonus mileage (r = 0.62), as well as between recent flight "
        "miles and flight transaction counts (r = 0.93), and bonus miles and bonus transactions (r = 0.90). Constant features (cc2_miles and cc3_miles) "
        "have zero variance post-filtering and are excluded from the correlation matrix to ensure a clean, fully populated structure. "
        "PCA revealed that the first two components capture 39.73% and 20.83% of total variance respectively (60.56% cumulative).",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/03_correlation_heatmap.png'):
        img = Image('figures/03_correlation_heatmap.png', width=11.5*cm, height=7*cm)
        story.append(img)
        story.append(Paragraph("Figure 3: Feature correlation matrix across log-transformed customer attributes", caption_style))

    story.append(PageBreak())

    # ================= 6. MODEL EVALUATION =================
    story.append(Paragraph("5. Model Evaluation and Algorithm Comparison", h1))
    story.append(Paragraph(
        "K-Means hyperparameter selection combined inertia elbow analysis and silhouette score evaluation across K=2 to 10. "
        "Inertia declined sharply from K=1 (28,984) to K=4 (12,568), demonstrating an elbow point at K=4. "
        "Silhouette scores reached an absolute maximum at K=4 (0.3199), confirming 4 clusters as the optimal partitioning choice.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/06_kmeans_elbow_silhouette.png'):
        img = Image('figures/06_kmeans_elbow_silhouette.png', width=14*cm, height=5*cm)
        story.append(img)
        story.append(Paragraph("Figure 4: K-Means evaluation: Inertia elbow curve (left) and Silhouette score profile (right) with K=4 highlighted", caption_style))

    story.append(Paragraph(
        "For Hierarchical clustering, truncated dendrograms were constructed across four linkage criteria. Ward linkage achieved the highest "
        "silhouette score (0.3317) at K=4 with clean, balanced branches. Single linkage exhibited severe chaining (silhouette = 0.2641), "
        "while Complete (0.2453) and Average (0.2518) produced less coherent partitions.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/07_dendrograms.png'):
        img = Image('figures/07_dendrograms.png', width=14*cm, height=6.5*cm)
        story.append(img)
        story.append(Paragraph("Figure 5: Truncated hierarchical clustering dendrograms across linkage criteria showing silhouette at K=4", caption_style))

    story.append(Paragraph(
        f"For DBSCAN, the 5-nearest neighbour distance curve sorted in ascending order demonstrated a clear knee region between distances 0.88 and 0.94. "
        f"Grid search across eps and min_samples identified eps = {metrics['best_dbscan_eps']} and min_samples = {metrics['best_dbscan_min_samples']} "
        f"as optimal, separating {metrics['dbscan']['n_noise']} accounts ({metrics['dbscan']['noise_pct']}%) as noise and identifying 3 dense customer cores "
        f"with non-noise silhouette = {metrics['dbscan']['silhouette']}.",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    # Model comparison table
    story.append(Paragraph("<b>Table 1: Clustering Algorithm Performance Comparison</b>", h2))
    table_rows = [
        [Paragraph("<b>Algorithm</b>", table_header), Paragraph("<b>Clusters</b>", table_header),
         Paragraph("<b>Noise Pts (%)</b>", table_header), Paragraph("<b>Silhouette</b>", table_header),
         Paragraph("<b>Davies-Bouldin</b>", table_header), Paragraph("<b>Calinski-Harabasz</b>", table_header)]
    ]
    for _, row in comp_df.iterrows():
        table_rows.append([
            Paragraph(str(row['Algorithm']), table_cell),
            Paragraph(str(int(row['N_Clusters'])), table_cell),
            Paragraph(f"{int(row['Noise_Points'])} ({row['Noise_Pct']}%)", table_cell),
            Paragraph(f"{row['Silhouette']:.4f}", table_cell),
            Paragraph(f"{row['Davies_Bouldin']:.4f}", table_cell),
            Paragraph(f"{row['Calinski_Harabasz']:.2f}", table_cell)
        ])
    t = Table(table_rows, colWidths=[3.8*cm, 1.6*cm, 2.4*cm, 2.2*cm, 2.7*cm, 3.3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1b365d")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ================= 7. INTERPRETATION & CLUSTERING PROFILES =================
    if os.path.exists('figures/09_cluster_pca_scatter.png'):
        img = Image('figures/09_cluster_pca_scatter.png', width=14.5*cm, height=4.8*cm)
        story.append(img)
        story.append(Paragraph("Figure 6: 2D PCA cluster visualizations with centroids (K-Means) and noise classification (DBSCAN)", caption_style))

    story.append(Paragraph("6. Cluster Profiling and Marketing Strategy", h1))
    story.append(Paragraph(
        "K-Means segments were profiled by unscaling cluster centroids back to original units and analyzing award redemption rates:",
        normal
    ))
    story.append(Spacer(1, 0.15*cm))

    if os.path.exists('figures/10_cluster_heatmap.png'):
        img = Image('figures/10_cluster_heatmap.png', width=14*cm, height=6.5*cm)
        story.append(img)
        story.append(Paragraph("Figure 7: Standardized cluster feature heatmap for K-Means customer personas", caption_style))

    story.append(Paragraph(
        "<b>Cluster 0: Active Frequent Flyers (1,005 members, 27.7%)</b><br/>"
        "Profile: High average balance (103,836 miles), 1,076 flight miles in the past 12 months, 3.03 flight transactions, "
        "and 54.2% award flight redemption rate. Strategy: Retain with tier upgrades, premium lounge access, and partner travel perks.<br/><br/>"
        "<b>Cluster 1: Credit Card Loyalty Earners (1,630 members, 45.0%)</b><br/>"
        "Profile: Substantial balance (75,965 miles), 20,900 bonus miles through co-branded credit cards (cc1 tier = 2.5), "
        "but zero flights recorded in the past year; 36.2% award rate. Strategy: Convert mileage into revenue flights via limited-time "
        "seat discounts and redemption bonus windows.<br/><br/>"
        "<b>Cluster 2: Inactive / Infrequent Members (987 members, 27.2%)</b><br/>"
        "Profile: Low account balances (22,418 miles), minimal transactions (1.79 bonus transactions), 10 flight miles, and only 11.8% award rate. "
        "Strategy: Low-cost automated email re-engagement campaigns with welcome-back bonus incentives to prevent churn.<br/><br/>"
        "<b>Cluster 3: Outlier Tier Account (1 member, <0.1%)</b><br/>"
        "Profile: Account with extraordinary qualification miles (250 Qual_miles, 8 flight transactions) isolated as a distinct boundary point.",
        normal
    ))

    # ================= 8. CONCLUSION & LIMITATIONS =================
    story.append(PageBreak())
    story.append(Paragraph("7. Conclusion and Limitations", h1))
    story.append(Paragraph(
        "This investigation demonstrated that customer segmentation on airline frequent flyer data is highly effective when combined with "
        "appropriate transformations. K-Means and Hierarchical Ward clustering yielded consistent, highly interpretable customer partitions. "
        "DBSCAN demonstrated utility in isolating 220 noise accounts with atypical usage patterns.",
        normal
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Key limitations include: (1) K selection retains an element of subjectivity despite elbow and silhouette validation, "
        "(2) clusters depend on log1p scaling and StandardScaler assumptions, (3) DBSCAN struggles with varying density across features, "
        "and (4) PCA 2D projections capture 60.6% of variance, omitting higher-dimensional variation.",
        normal
    ))

    # ================= 9. REFERENCES & ATTRIBUTION =================
    story.append(Paragraph("8. References & Attribution", h1))
    story.append(Paragraph(
        "1. Shmueli, G., and Bruce, P. C. (2016). <i>Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python</i>. John Wiley & Sons. East-West Airlines dataset.<br/>"
        "2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.<br/>"
        f"3. GitHub Repository: {github_url}<br/><br/>"
        f"<b>Attribution:</b> Code structure drafted with AI assistance (Claude); executed, verified and interpreted by {student_name}. "
        "Dataset source: Shmueli and Bruce (2016), East-West Airlines frequent flyer dataset.",
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
