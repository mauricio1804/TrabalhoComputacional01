import cv2 as cv
import numpy as np

class filtros:
    def _to_gray(self, img):
        if len(img.shape) == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    def _binarizar(self, img):
        gray = self._to_gray(img)
        _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        return binary

    def _get_kernel(self, tipo='RECT', tamanho=5):
        if tipo == 'RECT':
            return cv.getStructuringElement(cv.MORPH_RECT, (tamanho, tamanho))
        elif tipo == 'ELLIPSE':
            return cv.getStructuringElement(cv.MORPH_ELLIPSE, (tamanho, tamanho))
        elif tipo == 'CROSS':
            return cv.getStructuringElement(cv.MORPH_CROSS, (tamanho, tamanho))
        else:
            return np.ones((tamanho, tamanho), np.uint8)

    def _apply_filters(self, app, bgr):
        """Aplica o filtro escolhido pelo usuário."""
        choice = app.effect_var.get()
        img = bgr.copy()

        # Nenhum filtro
        if choice == "Nenhum":
            return cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Conversão para cinza
        elif choice == "Cinza":
            gray = self._to_gray(img)
            return cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

        # Negativo
        elif choice == "Negativo":
            neg = 255 - img
            return cv.cvtColor(neg, cv.COLOR_BGR2RGB)

        # Binarização Otsu
        elif choice == "Otsu":
            gray = self._to_gray(img)
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            return cv.cvtColor(binary, cv.COLOR_GRAY2RGB)

        # Erosão
        elif choice == "Erosão":
            binary = self._binarizar(img)
            kernel = self._get_kernel('RECT', 5)
            eroded = cv.erode(binary, kernel)
            return cv.cvtColor(eroded, cv.COLOR_GRAY2RGB)

        # Dilatação
        elif choice == "Dilatação":
            binary = self._binarizar(img)
            kernel = self._get_kernel('RECT', 5)
            dilated = cv.dilate(binary, kernel)
            return cv.cvtColor(dilated, cv.COLOR_GRAY2RGB)

        # Abertura
        elif choice == "Abertura":
            binary = self._binarizar(img)
            kernel = self._get_kernel('ELLIPSE', 5)
            eroded = cv.erode(binary, kernel)
            opened = cv.dilate(eroded, kernel)
            return cv.cvtColor(opened, cv.COLOR_GRAY2RGB)

        # Fechamento
        elif choice == "Fechamento":
            binary = self._binarizar(img)
            kernel = self._get_kernel('ELLIPSE', 5)
            dilated = cv.dilate(binary, kernel)
            closed = cv.erode(dilated, kernel)
            return cv.cvtColor(closed, cv.COLOR_GRAY2RGB)

        # Detector de Bordas (Canny)
        elif choice == "Detector de Bordas (Canny)":
            gray = self._to_gray(img)
            edges = cv.Canny(gray, 100, 200)
            return cv.cvtColor(edges, cv.COLOR_GRAY2RGB)

        # Suavização média
        elif choice == "Suavização (Média)":
            blur = cv.blur(img, (5, 5))
            return cv.cvtColor(blur, cv.COLOR_BGR2RGB)

        # Suavização mediana
        elif choice == "Suavização (Mediana)":
            blur = cv.medianBlur(img, 5)
            return cv.cvtColor(blur, cv.COLOR_BGR2RGB)

        # Fallback
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
