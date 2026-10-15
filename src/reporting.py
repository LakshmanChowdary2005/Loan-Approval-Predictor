"""
Automated PDF and Excel Report Generation Module.
Generates institutional underwriting evaluation documents using ReportLab
and formatted analytical spreadsheets using OpenPyXL.
"""
import io
import os
from datetime import datetime
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_prediction_pdf(record):
    """
    Builds a professional executive PDF underwriting report for a single prediction.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0B192C')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#64748B')
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E3E62'),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748B')
    )

    story = []

    # Header Title
    story.append(Paragraph("CREDITPULSE AI - UNDERWRITING DECISION SUPPORT REPORT", title_style))
    story.append(Spacer(1, 4))
    gen_time = datetime.now().strftime('%B %d, %Y - %H:%M:%S UTC')
    pred_id = record.get('prediction_id', 'N/A')
    model_name = record.get('model_name', 'Random Forest')
    story.append(Paragraph(f"Application ID: <b>{pred_id}</b> | Generated: {gen_time} | Engine: {model_name}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#00ADB5'), spaceAfter=15))

    # Executive Verdict Banner
    decision = record.get('predicted_decision', 'PENDING')
    confidence = record.get('confidence', 0.0)
    prob_app = record.get('prob_approved', 0.0)
    prob_rej = record.get('prob_rejected', 0.0)
    risk_tier = record.get('risk_tier', 'MODERATE')

    banner_color = colors.HexColor('#10B981') if decision == 'APPROVED' else colors.HexColor('#EF4444')
    banner_text = f"<b>PREDICTION: {decision}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Model Confidence: {confidence}% (Approval Prob: {prob_app}% / Rejection Prob: {prob_rej}%) &nbsp;&nbsp;|&nbsp;&nbsp; Risk Tier: {risk_tier}"

    banner_data = [[Paragraph(f"<font color='white' size=11>{banner_text}</font>", body_style)]]
    banner_table = Table(banner_data, colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 15))

    # Applicant Profile & Input Parameters Table
    story.append(Paragraph("1. Submitted Applicant Profile", section_style))
    applicant_data = [
        ["Gender:", str(record.get('Gender', 'Male')), "Applicant Income:", f"${float(record.get('ApplicantIncome', 0)):,.2f}"],
        ["Marital Status:", str(record.get('Married', 'Yes')), "Co-Applicant Income:", f"${float(record.get('CoapplicantIncome', 0)):,.2f}"],
        ["Dependents:", str(record.get('Dependents', '0')), "Requested Loan:", f"${float(record.get('LoanAmount', 0))*1000:,.2f}"],
        ["Education:", str(record.get('Education', 'Graduate')), "Repayment Term:", f"{float(record.get('Loan_Amount_Term', 360)):.0f} Months"],
        ["Self Employed:", str(record.get('Self_Employed', 'No')), "Credit History:", "1.0 (Prime / Meets Criteria)" if float(record.get('Credit_History', 1)) == 1 else "0.0 (Unfavorable / Defaulter)"],
        ["Property Area:", str(record.get('Property_Area', 'Urban')), "Total Monthly Income:", f"${float(record.get('ApplicantIncome', 0)) + float(record.get('CoapplicantIncome', 0)):,.2f}"]
    ]
    app_table = Table(applicant_data, colWidths=[110, 160, 130, 140])
    app_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E293B')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(app_table)
    story.append(Spacer(1, 15))

    # Financial Affordability & Risk Ratios Table
    story.append(Paragraph("2. Financial Risk & Debt Affordability Analysis", section_style))
    dti = record.get('dti_ratio', 0.0)
    emi = record.get('emi_monthly', 0.0)
    financial_data = [
        ["Estimated Monthly Installment (EMI):", f"${float(emi):,.2f}/month", "Debt-To-Income (DTI) Ratio:", f"{dti:.2f}%"],
        ["Assumed APR Rate:", "8.50% Fixed", "Safe DTI Guideline:", "36.00% Max Threshold"],
        ["Borrowing Stress Tier:", f"{risk_tier} Risk", "Collateral Demographic Liquidity:", str(record.get('Property_Area', 'Urban'))]
    ]
    fin_table = Table(financial_data, colWidths=[170, 100, 160, 110])
    fin_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(fin_table)
    story.append(Spacer(1, 15))

    # Explainable AI Insights
    story.append(Paragraph("3. Explainable AI Decision Attributions", section_style))
    shap_data = record.get('shap_summary', {})
    nl_summary = shap_data.get('natural_language_explanation', '') if isinstance(shap_data, dict) else ''
    if not nl_summary:
        ch = float(record.get('Credit_History', 1))
        nl_summary = (
            f"The primary driver behind this {decision.lower()} prediction is Credit History ({ch:.1f}). "
            f"Applicants with prime credit records exhibit the highest statistical probability of automated approval, "
            f"supplemented by the applicant's income and debt-to-income stability."
        )

    story.append(Paragraph(f"<b>Statistical Attribution Summary:</b> {nl_summary}", body_style))
    story.append(Spacer(1, 15))

    # Regulatory Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
    story.append(Paragraph(
        "<b>DECISION SUPPORT DISCLAIMER:</b> This automated evaluation is generated by supervised machine learning models "
        "trained on historical loan application datasets. It serves strictly as an analytical decision-support and risk-awareness "
        "tool for underwriters. This output does not constitute an official commitment to lend, credit rating, or binding legal "
        "agreement. All real-world loan applications require independent compliance with fair lending statutes and manual verification.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_prediction_excel(record):
    """
    Builds a formatted Excel spreadsheet (.xlsx) for an individual prediction using OpenPyXL.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Underwriting Summary"

    # Styling Palettes
    navy_header = PatternFill(start_color="0B192C", end_color="0B192C", fill_type="solid")
    teal_fill = PatternFill(start_color="00ADB5", end_color="00ADB5", fill_type="solid")
    light_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    white_bold = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    navy_bold = Font(name="Calibri", size=11, bold=True, color="0B192C")
    title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    regular_font = Font(name="Calibri", size=10, color="1E293B")
    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    # Title Banner
    ws.merge_cells("A1:D1")
    title_cell = ws["A1"]
    title_cell.value = "CREDITPULSE AI - LOAN DECISION SUPPORT REPORT"
    title_cell.fill = navy_header
    title_cell.font = title_font
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    # Meta Row
    pred_id = record.get('prediction_id', 'N/A')
    model_name = record.get('model_name', 'Random Forest')
    ws["A2"] = "Application ID:"
    ws["B2"] = pred_id
    ws["C2"] = "Generated Timestamp:"
    ws["D2"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for c in ["A2", "C2"]:
        ws[c].font = navy_bold
    ws["A2"].fill = light_fill
    ws["C2"].fill = light_fill

    # Decision Banner
    ws.merge_cells("A4:D4")
    decision = record.get('predicted_decision', 'PENDING')
    dec_cell = ws["A4"]
    dec_cell.value = f"PREDICTED OUTCOME: {decision} | Confidence: {record.get('confidence', 0)}% | Risk Tier: {record.get('risk_tier', 'MODERATE')}"
    dec_cell.fill = PatternFill(start_color="10B981" if decision == "APPROVED" else "EF4444", fill_type="solid")
    dec_cell.font = white_bold
    dec_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[4].height = 24

    # Section 1 Header
    ws.merge_cells("A6:D6")
    sec1 = ws["A6"]
    sec1.value = "1. APPLICANT INPUT PARAMETERS"
    sec1.fill = teal_fill
    sec1.font = white_bold

    fields = [
        ("Gender", record.get('Gender', 'Male'), "Applicant Income ($)", float(record.get('ApplicantIncome', 0))),
        ("Marital Status", record.get('Married', 'Yes'), "Co-Applicant Income ($)", float(record.get('CoapplicantIncome', 0))),
        ("Dependents", str(record.get('Dependents', '0')), "Requested Loan ($k)", float(record.get('LoanAmount', 0))),
        ("Education", record.get('Education', 'Graduate'), "Term (Months)", float(record.get('Loan_Amount_Term', 360))),
        ("Self Employed", record.get('Self_Employed', 'No'), "Credit History", float(record.get('Credit_History', 1.0))),
        ("Property Area", record.get('Property_Area', 'Urban'), "Model Algorithm", model_name)
    ]

    r = 7
    for f in fields:
        ws[f"A{r}"] = f[0]
        ws[f"B{r}"] = f[1]
        ws[f"C{r}"] = f[2]
        ws[f"D{r}"] = f[3]
        ws[f"A{r}"].font = navy_bold
        ws[f"C{r}"].font = navy_bold
        for col in ["A", "B", "C", "D"]:
            ws[f"{col}{r}"].border = thin_border
        r += 1

    # Section 2 Header
    r += 1
    ws.merge_cells(f"A{r}:D{r}")
    sec2 = ws[f"A{r}"]
    sec2.value = "2. FINANCIAL RISK & AFFORDABILITY RATIOS"
    sec2.fill = teal_fill
    sec2.font = white_bold

    r += 1
    fin_rows = [
        ("Monthly EMI ($)", float(record.get('emi_monthly', 0)), "Debt-To-Income (DTI %)", float(record.get('dti_ratio', 0))),
        ("Approval Probability (%)", float(record.get('prob_approved', 0)), "Rejection Probability (%)", float(record.get('prob_rejected', 0))),
        ("Benchmark Safe DTI", "36.00%", "Assumed Fixed APR", "8.50%")
    ]
    for row in fin_rows:
        ws[f"A{r}"] = row[0]
        ws[f"B{r}"] = row[1]
        ws[f"C{r}"] = row[2]
        ws[f"D{r}"] = row[3]
        ws[f"A{r}"].font = navy_bold
        ws[f"C{r}"].font = navy_bold
        for col in ["A", "B", "C", "D"]:
            ws[f"{col}{r}"].border = thin_border
        r += 1

    # Set Column widths
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['C'].width = 25
    ws.column_dimensions['D'].width = 25

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

def generate_batch_excel_report(batch_results):
    """
    Builds a downloadable batch scoring Excel spreadsheet from processed CSV rows.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Batch Scoring Results"

    navy_header = PatternFill(start_color="0B192C", end_color="0B192C", fill_type="solid")
    white_bold = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    headers = [
        "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Term", "Credit_History",
        "Property_Area", "Prediction", "Decision", "Confidence (%)", "Approval_Prob (%)", "Rejection_Prob (%)", "Risk_Tier"
    ]

    ws.append(headers)
    for col_num, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = navy_header
        cell.font = white_bold
        cell.alignment = Alignment(horizontal="center")

    for rec in batch_results:
        row = [
            rec.get('Loan_ID', 'N/A'),
            rec.get('Gender', 'Male'),
            rec.get('Married', 'Yes'),
            rec.get('Dependents', '0'),
            rec.get('Education', 'Graduate'),
            rec.get('Self_Employed', 'No'),
            float(rec.get('ApplicantIncome', 0)),
            float(rec.get('CoapplicantIncome', 0)),
            float(rec.get('LoanAmount', 0)),
            float(rec.get('Loan_Amount_Term', 360)),
            float(rec.get('Credit_History', 1.0)),
            rec.get('Property_Area', 'Urban'),
            int(rec.get('Prediction', 1)),
            rec.get('Decision', 'APPROVED'),
            float(rec.get('Confidence', 0.0)),
            float(rec.get('Approved_Prob', 0.0)),
            round(100.0 - float(rec.get('Approved_Prob', 0.0)), 2),
            rec.get('Risk_Tier', 'MODERATE')
        ]
        ws.append(row)

    # Format widths
    for col in ws.columns:
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = 16

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer
