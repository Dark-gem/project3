import openpyxl
import os

class ExcelHandler:
    def __init__(self, workbook_path='employees.xlsx'):
        self.workbook_path = workbook_path
        self._ensure_workbook_exists()

    def _ensure_workbook_exists(self):
        """Check if the workbook exists; if not, create it."""
        if not os.path.exists(self.workbook_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Employees"
            ws.append(["Name"])  # Add a header row
            wb.save(self.workbook_path)
        self.wb = openpyxl.load_workbook(self.workbook_path)
        self.ws = self.wb.active

    def add_employee(self, name):
        """Add a new employee's name to the Excel file."""
        self.ws.append([name])
        self.wb.save(self.workbook_path)

    def generate_report(self):
        """Generate a report as needed (optional, for future use)."""
        # Implement report generation logic if required
        pass
