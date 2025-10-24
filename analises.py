import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class Analise:
    def __init__(self, app):
        self.app = app 

    def run_analysis(self):
        source_bgr = None
        if self.app.running:
            with self.app.lock:
                source_bgr = self.app.frame.copy() if self.app.frame is not None else None
        else:
            source_bgr = self.app.image_bgr.copy() if self.app.image_bgr is not None else None

        if source_bgr is None:
            tk.messagebox.showinfo("Info", "Nenhuma imagem disponível para análise.")
            return

        choice = self.app.analysis_var.get()
        self.app.results_text.delete("1.0", tk.END)

        gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        binary = self.app.filtros._ensure_binary(source_bgr)

        if choice == "Histograma (Tonais ou Binário)":
            self.app.results_text.insert(tk.END, "Histograma gerado (escala de cinza e binário).\n")
            self._show_histograms(gray, binary)

        elif choice == "Área (pixels brancos)":
            area = int(np.count_nonzero(binary == 255))
            self.app.results_text.insert(tk.END, f"Área (pixels brancos): {area}\n")

        elif choice == "Perímetro (contorno)":
            perimeter = self._compute_perimeter(binary)
            self.app.results_text.insert(tk.END, f"Perímetro total: {perimeter:.2f}\n")

        elif choice == "Diâmetro (máx. distância)":
            diameter = self._compute_diameter(binary)
            self.app.results_text.insert(tk.END, f"Diâmetro máximo: {diameter:.2f} px\n")

        elif choice == "Contagem de objetos (crescimento de região)":
            count, labeled = self._region_growth_count(binary)
            self.app.results_text.insert(tk.END, f"Objetos encontrados: {count}\n")
            self._show_label_overlay(source_bgr, labeled)

        else:
            self.app.results_text.insert(tk.END, "Selecione uma análise válida.\n")

    def _histogram_image(self, img_gray):
        hist_size = 256
        hist = cv2.calcHist([img_gray], [0], None, [hist_size], [0, 256]).flatten()
        hist_norm = hist / (hist.max() if hist.max() > 0 else 1)
        w, h = 512, 200
        hist_img = np.full((h, w, 3), 255, dtype=np.uint8)
        for i in range(hist_size):
            x = int(i * (w / hist_size))
            y = int(hist_norm[i] * (h - 20))
            cv2.line(hist_img, (x, h-1), (x, h-1 - y), (50, 50, 220), 1)
        cv2.rectangle(hist_img, (0, 0), (w-1, h-1), (0, 0, 0), 1)
        cv2.putText(hist_img, "Histograma", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return hist_img

    def _show_histograms(self, gray, binary):
        hist_gray_bgr = self._histogram_image(gray)
        hist_bin_bgr = self._histogram_image(binary)
        self._show_image(hist_gray_bgr, "Histograma - Tons de Cinza")
        self._show_image(hist_bin_bgr, "Histograma - Binário")

    def _compute_perimeter(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return sum(cv2.arcLength(c, True) for c in contours)

    def _compute_diameter(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_d = 0.0
        for c in contours:
            hull = cv2.convexHull(c)
            pts = hull.reshape(-1, 2)
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d > max_d:
                        max_d = d
        return max_d

    def _region_growth_count(self, binary):
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=np.uint8)
        label_img = np.zeros((h, w), dtype=np.int32)
        label = 0
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for y in range(h):
            for x in range(w):
                if binary[y, x] == 255 and visited[y, x] == 0:
                    label += 1
                    queue = [(y, x)]
                    visited[y, x] = 1
                    label_img[y, x] = label
                    while queue:
                        cy, cx = queue.pop(0)
                        for dy, dx in neighbors:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary[ny, nx] == 255 and visited[ny, nx] == 0:
                                    visited[ny, nx] = 1
                                    label_img[ny, nx] = label
                                    queue.append((ny, nx))
        return label, label_img

    def _show_label_overlay(self, bgr, labels):
        max_label = labels.max()
        if max_label <= 0:
            self.app.results_text.insert(tk.END, "Nenhum objeto para visualizar.\n")
            return
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 255, size=(max_label+1, 3), dtype=np.uint8)
        overlay = np.zeros_like(bgr)
        h, w = labels.shape
        for y in range(h):
            for x in range(w):
                overlay[y, x] = colors[labels[y, x]]
        blend = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0.0)
        self._show_image(blend, "Objetos rotulados (overlay)")

    def _show_image(self, bgr, title):
        top = tk.Toplevel(self.app.root)
        top.title(title)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tkimg = ImageTk.PhotoImage(pil)
        lbl = tk.Label(top, image=tkimg)
        lbl.image = tkimg
        lbl.pack()
