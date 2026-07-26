from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QHeaderView,
    QAbstractItemView, QTableWidgetItem)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QBrush
from core.mold_advisor import get_advice_table


class AdviceTableWidget(QWidget):
    point_highlighted = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#", "\u89d2\u5ea6(\u00b0)", "\u504f\u5dee", "X", "Y", "\u4fee\u6a21\u5efa\u8bae"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1e1e36; color: #e8e8f0;
                           gridline-color: #333355; font-size: 9pt;
                           border: none; border-radius: 4px; }
            QHeaderView::section { background-color: #252542; color: #9898b8;
                                   padding: 4px; border: 1px solid #333355;
                                   font-size: 9pt; }
            QTableWidget::item:selected { background-color: #3b82f6; color: white; }
        """)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table)

    def update_data(self, polar_points):
        rows = get_advice_table(polar_points)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(row["no"])))
            self.table.setItem(i, 1, QTableWidgetItem(row["angle"]))
            item_delta = QTableWidgetItem(row["delta_r"])
            val = float(row["delta_r"])
            if val > 0:
                item_delta.setForeground(QBrush(QColor("#f87171")))
            else:
                item_delta.setForeground(QBrush(QColor("#4ade80")))
            item_delta.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, item_delta)
            self.table.setItem(i, 3, QTableWidgetItem(row["x"]))
            self.table.setItem(i, 4, QTableWidgetItem(row["y"]))
            self.table.setItem(i, 5, QTableWidgetItem(row["advice"]))

    def _on_selection(self):
        rows = self.table.selectedIndexes()
        if rows:
            self.point_highlighted.emit(rows[0].row())
