from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QPlainTextEdit, QLabel,
    QFileDialog, QMessageBox, QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QClipboard, QFont
from PyQt5.QtWidgets import QApplication
from core.data_loader import parse_clipboard_text, load_csv
import re


class DataImportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self.btn_paste = QPushButton("\U0001f4cb 输入数据")
        self.btn_paste.clicked.connect(self._show_input_dialog)
        self.btn_csv = QPushButton("\U0001f4c2 导入 CSV")
        self.btn_csv.clicked.connect(self._load_csv)
        for b in [self.btn_paste, self.btn_csv]:
            b.setStyleSheet("""
                QPushButton { background: #252542; color: #e8e8f0;
                              border: none; padding: 5px 12px;
                              border-radius: 4px; font-size: 9pt; }
                QPushButton:hover { background: #2d2d50; }
            """)
        btn_row.addWidget(self.btn_paste)
        btn_row.addWidget(self.btn_csv)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.info_label = QLabel("尚未导入数据")
        self.info_label.setStyleSheet("color: #9898b8; font-size: 9pt; padding: 2px 0;")
        layout.addWidget(self.info_label)

    def _show_input_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("输入坐标数据")
        dialog.resize(540, 460)
        dialog.setStyleSheet("QDialog { background-color: #1a1a2e; }")
        layout = QVBoxLayout(dialog)
        layout.setSpacing(6)

        hint = QLabel(
            "\u2750 每行输入 X 和 Y，用 Tab / \u9017\u53f7 / \u7a7a\u683c / \u5206\u53f7 \u5206\u9694\n"
            "\u2750 \u652f\u6301 Ctrl+Z \u64a4\u9500\u3001Ctrl+Shift+Z \u91cd\u505a\u3001Ctrl+V \u7c98\u8d34"
        )
        hint.setStyleSheet("color: #9898b8; font-size: 9pt; padding: 2px 0;")
        layout.addWidget(hint)

        editor = QPlainTextEdit()
        editor.setStyleSheet("""
            QPlainTextEdit { background-color: #12121e; color: #e8e8f0;
                             border: 1px solid #333355; border-radius: 4px;
                             font-family: Consolas; font-size: 11pt;
                             padding: 8px; selection-background-color: #3b82f6; }
        """)
        editor.setTabStopDistance(24)
        editor.setPlaceholderText(
            "100.5  200.3\n150.2  180.7\n120.0  210.5\n..."
        )
        layout.addWidget(editor, 1)

        # Preview bar
        preview = QLabel("")
        preview.setStyleSheet("color: #9898b8; font-size: 9pt; padding: 2px 0;")
        layout.addWidget(preview)

        def _update_preview():
            text = editor.toPlainText()
            lines = text.strip().splitlines()
            count = 0
            for line in lines:
                parts = re.split(r'[\t,; ]+', line.strip())
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        break
                if len(nums) >= 2:
                    count += 1
            preview.setText(f"\u6709\u6548\u70b9\u6570: {count}" if count >= 3
                           else f"\u6709\u6548\u70b9\u6570: {count} (\u9700\u22653)")
            preview.setStyleSheet(
                "color: #4ade80; font-size: 9pt; padding: 2px 0;" if count >= 3
                else "color: #fbbf24; font-size: 9pt; padding: 2px 0;"
            )

        editor.textChanged.connect(_update_preview)
        _update_preview()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel(""))
        btn_row.addStretch()
        btn_clear = QPushButton("\u6e05\u7a7a")
        btn_clear.setStyleSheet("""
            QPushButton { background: #6b6b8a; color: white;
                          border: none; padding: 5px 14px;
                          border-radius: 4px; font-size: 9pt; }
            QPushButton:hover { background: #555577; }
        """)
        btn_clear.clicked.connect(editor.clear)
        btn_close = QPushButton("\u53d6\u6d88")
        btn_close.setStyleSheet("""
            QPushButton { background: #6b6b8a; color: white;
                          border: none; padding: 5px 14px;
                          border-radius: 4px; font-size: 9pt; }
            QPushButton:hover { background: #555577; }
        """)
        btn_close.clicked.connect(dialog.reject)
        btn_apply = QPushButton("\u786e\u5b9a")
        btn_apply.setStyleSheet("""
            QPushButton { background: #3b82f6; color: white;
                          border: none; padding: 5px 20px;
                          border-radius: 4px; font-size: 9pt; font-weight: bold; }
            QPushButton:hover { background: #2563eb; }
        """)
        btn_apply.clicked.connect(dialog.accept)
        btn_row.addWidget(btn_clear)
        btn_row.addWidget(btn_close)
        btn_row.addWidget(btn_apply)
        layout.addLayout(btn_row)

        if dialog.exec_() == QDialog.Accepted:
            text = editor.toPlainText()
            lines = text.strip().splitlines()
            pts = []
            for line in lines:
                parts = re.split(r'[\t,; ]+', line.strip())
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        break
                if len(nums) >= 2:
                    pts.append((nums[0], nums[1]))
            from models.data_models import MeasurePoint
            self.points = [MeasurePoint(x, y) for x, y in pts]
            self._update_info()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV (*.csv);;All (*.*)")
        if path:
            try:
                self.points = load_csv(path)
                self._update_info()
            except Exception as e:
                QMessageBox.warning(self, "导入失败", str(e))

    def _update_info(self):
        n = len(self.points)
        if n >= 3:
            self.info_label.setText(f"已导入 {n} 个测量点 \u2713")
            self.info_label.setStyleSheet("color: #4ade80; font-size: 9pt; padding: 2px 0;")
        elif n > 0:
            self.info_label.setText(f"已导入 {n} 个点（至少需要 3 个）")
            self.info_label.setStyleSheet("color: #fbbf24; font-size: 9pt; padding: 2px 0;")
        else:
            self.info_label.setText("未解析到有效坐标")
            self.info_label.setStyleSheet("color: #f87171; font-size: 9pt; padding: 2px 0;")
