import cv2
import numpy as np

class filtros:
    def _ensure_binary(self, img):
        """Converte imagem para binária usando Otsu."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _apply_filters(self, app, bgr):
        choice = app.effect_var.get()
        img = bgr.copy()

        if choice == "Nenhum":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif choice == "Cinza":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif choice == "Negativo":
            neg = 255 - img
            return cv2.cvtColor(neg, cv2.COLOR_BGR2RGB)
        elif choice == "Otsu":
            binary = self._ensure_binary(img)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        elif choice == "Suavização (Média)":
            blur = cv2.blur(img, (5, 5))
            return cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        elif choice == "Suavização (Mediana)":
            blur = cv2.medianBlur(img, 5)
            return cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        elif choice == "Detector de Bordas (Canny)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif choice in ["Erosão", "Dilatação", "Abertura", "Fechamento"]:
            binary = self._ensure_binary(img)
            kernel = np.ones((5, 5), np.uint8)
            if choice == "Erosão":
                result = cv2.erode(binary, kernel, iterations=1)
            elif choice == "Dilatação":
                result = cv2.dilate(binary, kernel, iterations=1)
            elif choice == "Abertura":
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            elif choice == "Fechamento":
                result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)