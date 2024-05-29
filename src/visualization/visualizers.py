from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.visualization.abstract import Visualizer


class DefaultPyPlotVisualizer(Visualizer):
    def __init__(self, mode: bool) -> None:
        super().__init__(mode)

    def visualize(self, model_name: str, figures: list[Figure]) -> None:
        for figure in figures:
            figure.show()


class PDFVisualizer(Visualizer):
    def __init__(self, mode: bool, pdf_path: str) -> None:
        super().__init__(mode)
        self.pdf_path = pdf_path

    def visualize(self, model_name: str, figures: list[Figure]) -> None:
        file_name = f'{self.pdf_path}/{model_name}.pdf'
        pdf = PdfPages(file_name)

        for figure in figures:
            figure.savefig(pdf, format='pdf')

        pdf.close()
