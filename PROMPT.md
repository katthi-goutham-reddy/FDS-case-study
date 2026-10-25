# Role
Act as a senior Data Science mentor and Python engineer. Help me complete my Fundamentals of Data Science (FDS) case studies 2 and 5 while making sure I understand the analysis and can explain and defend it in my faculty evaluation.

# Context
I am a B.E. CSE student (III Year, V Sem) at CBIT. I must submit, for EACH case study: an executed Jupyter Notebook, a PDF report, and the code pushed to one GitHub repository. CLAUDE.md in this folder is the authoritative spec: the faculty's guidelines, title-page fields, the task list for each case study, file names, folder layout, plotting rules and the PDF report structure. Follow it exactly. If anything in this prompt conflicts with CLAUDE.md, CLAUDE.md wins.

- CS-2: Clustering Analysis (K-Means, Hierarchical, DBSCAN) on East-West Airlines customers. Folder 2/. Brief: 2/Clustering Analysis.docx. Data: 2/EastWestAirlines.xlsx (sheet "data").
- CS-5: Wine Quality, statistics + multiple linear regression. Folder 5/. Brief: 5/Wine_Quality.docx. Data: 5/wine+quality.zip (red wine file, sep=';').

# Workflow (one case study at a time, CS-2 first, then CS-5)
## Phase 0: Inspect and plan
Read CLAUDE.md, both briefs and the datasets. Set up the venv and install requirements.txt. Then give me a concise plan (max ~15 lines per case study) mapping every task in the brief to a notebook section. Do not add analyses the brief or CLAUDE.md do not ask for. Ask me for the title-page details marked <PLACEHOLDER> in CLAUDE.md (name, roll no, section, faculty) and any other missing essentials. Then continue without waiting unless something is genuinely blocked.

## Phase 1: Build the notebook
Follow the CLAUDE.md task list for that case study. Purpose markdown before each code cell, Observation markdown after each output. Pipelines so nothing leaks from test data. Every plot titled, axis-labelled, saved to figures/ and interpreted. Key results saved to results/.

## Phase 2: Execute and verify
Run the notebook top to bottom with nbconvert --execute --inplace. Fix errors. Then rewrite every Observation, interpretation and conclusion cell from the actual outputs, and re-execute to confirm it is consistent. Check CS-2 cluster choices are justified by the elbow/silhouette/k-distance evidence, and that CS-5 scaling is fit on training data only and multicollinearity is discussed before coefficients are interpreted. Never invent, hardcode or guess results, and never substitute synthetic data.

## Phase 3: Teach me
After each case study, give me a short defence brief (bullet points): what each step does and why, the key numbers and what they mean, the key choices (K, linkage, eps/minPts, dataset, metrics) and why, likely viva questions with short answers.

## Phase 4: PDF reports
Build both PDFs exactly as the "PDF reports" section of CLAUDE.md says (reportlab script reading results/ and figures/, prescribed structure, hand-drawn style methodology diagram). Render the pages to images and check them yourself: readable figures, no overflow, no placeholders, numbers match the notebook. Ask me for the GitHub repo URL before the final build if I have not given it.

## Phase 5: Repository
Write README.md (objective of each case study, setup, how to run notebooks, and the report scripts, key findings). Pin requirements.txt to installed versions. git init and commit locally, then STOP and ask me to review before creating the GitHub repo or pushing. Only push after I approve, and never claim a push succeeded unless the command output shows it did.

# Working rules
- Simple, explainable methods suitable for a college submission. No unnecessary libraries.
- Plain student voice in all markdown and report text. No em dashes.
- No fabricated data, metrics, plots, references or outputs. No TODOs or placeholders left in final deliverables.
- Keep progress updates short.

# Final acceptance checklist (verify before saying done)
- Both notebooks run top to bottom with outputs saved in the .ipynb.
- Every task in both briefs and every required graph/table is present and interpreted.
- Both PDFs follow the prescribed structure, no placeholders remain, and all numbers match the notebooks.
- README and pinned requirements.txt are present; git status is clean after the commit.
- GitHub URL verified (only if I authorised the push).

Start with Phase 0.
