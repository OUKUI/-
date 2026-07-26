import os
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from models.data_models import AnalysisResult
from core.mold_advisor import get_advice_table

DARK_BG = HexColor("#1a1a2e")
LIGHT_TEXT = HexColor("#e8e8f0")

# Register Chinese font
_CJK_REGISTERED = False
def _register_cjk():
    global _CJK_REGISTERED
    if _CJK_REGISTERED:
        return
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simsun.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", fp))
                _CJK_REGISTERED = True
                return
            except Exception:
                continue
    # Fallback: try to find any CJK font
    import glob
    patterns = [
        "C:/Windows/Fonts/msyh*",
        "C:/Windows/Fonts/simsun*",
        "C:/Windows/Fonts/deng*",
        "C:/Windows/Fonts/yahei*",
    ]
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", fp))
                _CJK_REGISTERED = True
                return
            except Exception:
                continue


_register_cjk()
CJK = "CJKFont" if _CJK_REGISTERED else "Helvetica"


def generate_report(result, chart_img_path, output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=12, fontName=CJK,
    )
    normal_style = ParagraphStyle(
        "CustomNormal", parent=styles["Normal"],
        fontSize=10, leading=14, fontName=CJK,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#3b82f6"),
        spaceBefore=12, spaceAfter=6, fontName=CJK,
    )

    elements = []

    # Title
    elements.append(Paragraph("圆度分析报告", title_style))
    elements.append(Paragraph(f"报告生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 6*mm))

    # Results summary
    elements.append(Paragraph("拟合结果", heading_style))
    d = result.to_dict()
    summary_lines = [
        f"特征类型: {result.feature_type.value}",
        f"圆心: ({d['cx']:.4f}, {d['cy']:.4f})",
        f"半径: {d['radius']:.4f} {result.unit}",
        f"点数: {d['n_points']}",
        f"圆度值: {d['roundness']:.4f} {result.unit}",
        f"均方根误差(RMSE): {d['rmse']:.6f} {result.unit}",
        f"最大正偏差: +{d['peak_error']:.4f} {result.unit}",
        f"最大负偏差: {d['valley_error']:.4f} {result.unit}",
    ]
    for line in summary_lines:
        elements.append(Paragraph(line, normal_style))
    elements.append(Spacer(1, 4*mm))

    # Chart
    if os.path.exists(chart_img_path):
        elements.append(Paragraph("极坐标偏差图", heading_style))
        from PIL import Image as PILImage
        with PILImage.open(chart_img_path) as pil_img:
            iw, ih = pil_img.size
        target_w = 140 * mm
        target_h = target_w * ih / iw
        if target_h > 100 * mm:
            target_h = 100 * mm
            target_w = target_h * iw / ih
        img = Image(chart_img_path, width=target_w, height=target_h)
        elements.append(img)
        elements.append(Spacer(1, 4*mm))

    # Advice table
    elements.append(Paragraph("修模建议表", heading_style))
    table_data = get_advice_table(result.polar_points)
    if table_data:
        cell_style = ParagraphStyle("CellStyle", fontName=CJK, fontSize=8, leading=10)
        tbl = [[Paragraph(c, cell_style) for c in ["#", "角度(\u00b0)", f"\u504f\u5dee({result.unit})", "X", "Y", "\u4fee\u6a21\u5efa\u8bae"]]]
        for row in table_data:
            tbl.append([Paragraph(str(row["no"]), cell_style),
                        Paragraph(row["angle"], cell_style),
                        Paragraph(row["delta_r"], cell_style),
                        Paragraph(row["x"], cell_style),
                        Paragraph(row["y"], cell_style),
                        Paragraph(row["advice"], cell_style)])
        t = Table(tbl, colWidths=[15*mm, 30*mm, 30*mm, 30*mm, 30*mm, 45*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#3b82f6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f0f4ff")]),
        ]))
        elements.append(t)

    doc.build(elements)
    return output_path


def generate_and_save(result, chart_img, output_path):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        chart_path = f.name
    import cv2
    cv2.imwrite(chart_path, chart_img)
    try:
        generate_report(result, chart_path, output_path)
    finally:
        if os.path.exists(chart_path):
            os.unlink(chart_path)
    return output_path
