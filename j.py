import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTableView,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
import psycopg2

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "Two"
DB_USER = "postgres"
DB_PASSWORD = "12345"


def make_dsn():
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


class Mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        central = QWidget()
        layout = QVBoxLayout(central)

        btn = QPushButton("Проверить БД (SELECT 1)")
        btn.clicked.connect(self.con_DB)
        layout.addWidget(btn)

        self.table = QTableView()
        layout.addWidget(self.table)

        self.setCentralWidget(central)

        self.load_groups_to_tableview(self.table)

        self.table.setGeometry(0, 0, 200, 200)

    def _error(self, title, text):
        m = QMessageBox(self)
        m.setIcon(QMessageBox.Icon.Critical)
        m.setWindowTitle(title)
        m.setText(text)
        m.exec()

    def con_DB(self):
        dsn = make_dsn()
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT * from groups;")
                        val = cur.fetchone()[0]
                    self.status.showMessage(f"OK, SELECT 1 -> {val}", 3000)
            finally:
                conn.close()
        except Exception as e:
            self._error("Ошибка БД", str(e))

    def load_groups_to_tableview(self, table_view: QTableView):
        dsn = make_dsn()
        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            try:
                model = QStandardItemModel()
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM groups;")
                    rows = cur.fetchall()
                    col_names = [desc[0] for desc in cur.description]
                model.setColumnCount(len(col_names))
                model.setHorizontalHeaderLabels(col_names)

                for row in rows:
                    items = [QStandardItem("" if v is None else str(v)) for v in row]
                    model.appendRow(items)

                table_view.setModel(model)

            finally:
                conn.close()
        except Exception as e:
            self._error("Ошибка БД", str(e))


def main():
    app = QApplication(sys.argv)
    window = Mainwindow()
    window.setGeometry(500, 500, 500, 700)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
