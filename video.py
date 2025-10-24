import cv2
import numpy as np
import pygame
import os

class VideoProcessor:
    def __init__(self):
        self.tracker = None
        self.tracking = False
        self.track_window = None
        self.music_loaded = False
        self.template = None
        self.music_path = None

        try:
            pygame.mixer.init()
        except Exception as e:
            print("Aviso: pygame.mixer.init() falhou:", e)

    def _create_tracker_instance(self, preferred="CSRT"):
        """Tenta criar tracker de forma compatível com várias builds do OpenCV.
        Retorna (tracker_instance, nome_str) ou (None, None) se não houver."""
        creators = []
        if hasattr(cv2, "TrackerCSRT_create"):
            creators.append((lambda: cv2.TrackerCSRT_create(), "TrackerCSRT_create"))
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            creators.append((lambda: cv2.legacy.TrackerCSRT_create(), "legacy.TrackerCSRT_create"))
        if hasattr(cv2, "TrackerKCF_create"):
            creators.append((lambda: cv2.TrackerKCF_create(), "TrackerKCF_create"))
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            creators.append((lambda: cv2.legacy.TrackerKCF_create(), "legacy.TrackerKCF_create"))

        for cfunc, name in creators:
            try:
                tr = cfunc()
                if tr is not None:
                    print(f"[VideoProcessor] Tracker criado: {name}")
                    return tr, name
            except Exception as e:
                print(f"[VideoProcessor] Tentativa de criar {name} falhou: {e}")
                continue
        return None, None


    def init_tracker(self, frame, bbox):
        # Valida frame
        if frame is None:
            print("init_tracker: frame é None.")
            return False
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Valida bbox
        try:
            bx, by, bw, bh = (int(v) for v in bbox)
        except Exception as e:
            print("init_tracker: bbox inválido:", bbox, e)
            return False

        if bw <= 0 or bh <= 0:
            print(f"init_tracker: bbox com tamanho inválido w={bw}, h={bh}")
            return False

        tr, name = self._create_tracker_instance()
        if tr is None:
            print("Aviso: nenhum tracker disponível na sua instalação do OpenCV.")
            self.tracker = None
            self.tracking = False
            return False

        try:
            ret = None
            try:
                ret = tr.init(frame, (bx, by, bw, bh))
            except TypeError:
                tr.init(frame, (bx, by, bw, bh))
                ret = True

            if ret is False:
                raise Exception("Tracker.init() retornou False")
            self.tracker = tr
            self.tracking = True
            print(f"[VideoProcessor] Tracker inicializado com sucesso ({name}). bbox={bx,by,bw,bh}")
            return True
        except Exception as e:
            print("Falha ao inicializar tracker:", e)
            # limpa estado
            self.tracker = None
            self.tracking = False
            return False


    def update_tracker(self, frame):
        """Atualiza rastreamento e desenha retângulo."""
        if self.tracking and self.tracker is not None:
            try:
                ok, bbox = self.tracker.update(frame)
            except Exception:
                ok = False
                bbox = None

            if ok and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                # valida se a bbox é coerente
                if w > 5 and h > 5 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Rastreando", (x, max(0, y-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # se o rastreamento saiu dos limites, interrompe
                    self.tracking = False
                    self.tracker = None
                    cv2.putText(frame, "Perdeu o objeto", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Falha no rastreamento", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame


    # def detect_drumstick(self, frame):
    #     """Detecta baqueta com base em cor, forma e posição."""
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     lower = np.array([10, 50, 100])
    #     upper = np.array([30, 180, 255])
    #     mask = cv2.inRange(hsv, lower, upper)

    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     for c in contours:
    #         area = cv2.contourArea(c)
    #         if area < 800:
    #             continue
    #         x, y, w, h = cv2.boundingRect(c)
    #         aspect_ratio = max(w, h) / float(min(w, h) + 1)
    #         if aspect_ratio > 4:
    #             if x > 20 and y > 20 and x + w < frame.shape[1] - 20 and y + h < frame.shape[0] - 20:
    #                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #                 cv2.putText(frame, "Baqueta detectada!", (x, y - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    #                 self.play_music()
    #                 break
    #     return frame

    def detect_purple_bottle(self, frame):
        """Detecta garrafa roxa e inicia rastreamento + música."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([125, 50, 50])
        upper = np.array([155, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / float(min(w, h) + 1)

            if 1.5 < aspect_ratio < 4.5:
                if x > 20 and y > 20 and x + w < frame.shape[1] - 20 and y + h < frame.shape[0] - 20:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 180), 2)
                    cv2.putText(frame, "Garrafa roxa detectada!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 180), 2)
                    self.play_music()
                    self.init_tracker(frame, (x, y, w, h))
                    break
        return frame


    def load_template(self, path):
        """Carrega imagem modelo para detecção."""
        try:
            self.template = cv2.imread(path)
            if self.template is None:
                print("Erro: template não encontrado ou inválido.")
            else:
                print("Template carregado com sucesso.")
        except Exception as e:
            print("Erro ao carregar template:", e)

    def detect_template(self, frame, match_threshold=0.8):
        """Detecta objeto com base em imagem modelo (template matching)."""
        if self.template is None:
            return frame
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            h, w = gray_template.shape
            result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= match_threshold)
            found = False
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                cv2.putText(frame, "Template detectado", (pt[0], pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                found = True
                break
            if found:
                self.play_music()
        except Exception as e:
            print("Erro em detect_template:", e)
        return frame

    def set_music(self, path):
        try:
            pygame.mixer.music.load(path)
            self.music_loaded = True
            self.music_path = path
            print("Música carregada com sucesso.")
        except Exception as e:
            print("Erro ao carregar música:", e)

    def play_music(self):
        """Toca música se carregada e não estiver tocando."""
        if not self.music_loaded:
            return
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        except Exception as e:
            print("Erro ao tocar música:", e)
