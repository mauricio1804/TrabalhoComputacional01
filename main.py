import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from video import VideoProcessor
from filtros import filtros
from interface import InterfaceBuilder
from analises import Analise
import time

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Processamento Interativo - OpenCV")
        self.root.geometry("900x650")

        # Componentes principais
        self.video_processor = VideoProcessor()
        self.filtros = filtros()
        self.analisador = Analise(self)

        # Estado da aplicação
        self.selecting_roi = False
        self.roi_start = None
        self.roi_rect_id = None
        self.roi_coords = None
        self.image_bgr = None
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.video_cap = None
        self.video_running = False
        self.video_frame = None
        self.video_path = None

        # Filtros
        self.var_gray = tk.BooleanVar(value=False)
        self.var_negative = tk.BooleanVar(value=False)
        self.var_otsu = tk.BooleanVar(value=False)

        # Interface
        ui = InterfaceBuilder(self, self.root)
        ui.build_menu()
        ui.build_controls()
        ui.build_canvas()
        ui.build_status()

        self.root.after(50, self._refresh_canvas)

    # ---------- Seleção de ROI ----------
    def select_roi(self):
        if self.frame is None:
            messagebox.showinfo("Info", "Inicie a câmera para selecionar ROI.")
            return
        self._update_status("Clique e arraste no vídeo para selecionar ROI.")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def _on_mouse_down(self, event):
        self.roi_start = (event.x, event.y)
        if self.roi_rect_id:
            self.canvas.delete(self.roi_rect_id)
            self.roi_rect_id = None

    def _on_mouse_drag(self, event):
        if self.roi_start:
            x0, y0 = self.roi_start
            x1, y1 = event.x, event.y
            if self.roi_rect_id:
                self.canvas.delete(self.roi_rect_id)
            self.roi_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="red", width=2, tag="roi"
            )

    def _on_mouse_up(self, event):
        if not self.roi_start:
            return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if self.frame is not None:
            img_h, img_w = self.frame.shape[:2]
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2

            x0, x1 = x0 - offset_x, x1 - offset_x
            y0, y1 = y0 - offset_y, y1 - offset_y
            x0, x1 = int(x0 / scale), int(x1 / scale)
            y0, y1 = int(y0 / scale), int(y1 / scale)

            x0 = max(0, min(x0, img_w))
            y0 = max(0, min(y0, img_h))
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))

            bbox = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if bbox[2] > 10 and bbox[3] > 10:
                self.roi_coords = bbox
                self.video_processor.init_tracker(self.frame, bbox)
                self._update_status(f"Rastreamento iniciado: {bbox}")
            else:
                self._update_status("ROI muito pequena. Selecione uma área maior.")

            if self.roi_rect_id:
                self.canvas.delete(self.roi_rect_id)
                self.roi_rect_id = None
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.roi_start = None


    def open_image(self):
        path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Todos os arquivos", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        img
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
            return
        self.image_bgr = img
        self._update_status(f"Imagem carregada: {path}")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=[("Todos os arquivos", "*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return
        self.video_cap = cap
        self.video_path = path
        ret, frame = cap.read()
        with self.lock:
            self.video_frame = frame if ret else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._update_status(f"Vídeo carregado: {path}")

    def load_template(self):
        path = filedialog.askopenfilename(title="Selecionar template",
            filetypes=[("Todos", "*.*")]
        )
        if path:
            self.video_processor.load_template(path)
            self._update_status(f"Template carregado: {path}")

    def load_music(self):
        path = filedialog.askopenfilename(title="Selecionar música",
            filetypes=[("Todos", "*.*")]
        )
        if path:
            self.video_processor.set_music(path)
            self._update_status(f"Música carregada: {path}")

    def start_video(self):
        if self.video_running:
            self._update_status("Vídeo já em reprodução.")
            return
        if self.video_cap is None and self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap or not self.video_cap.isOpened():
            messagebox.showinfo("Info", "Abra um vídeo antes de iniciar.")
            return
        self.video_running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        self._update_status("Reprodução de vídeo iniciada.")

    def stop_video(self):
        self.video_running = False
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = None
        self.video_frame = None
        self._update_status("Vídeo parado.")

    def _video_loop(self):
        cap = self.video_cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time = 1.0 / fps
        while self.video_running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            with self.lock:
                self.video_frame = frame
            sleep_time = frame_time - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        cap.release()
        self.video_running = False

    def start_camera(self):
        if self.running:
            self._update_status("Câmera já em execução.")
            return
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Não foi possível acessar a câmera.")
            self.running = True
            threading.Thread(target=self._camera_loop, daemon=True).start()
            self._update_status("Câmera iniciada.")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        self._update_status("Câmera parada.")

    def _camera_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame
            cv2.waitKey(1)

    def run_analysis(self):
        self.analisador.run_analysis()

    def save_result(self):
        img = self._get_current_processed_image()
        if img is None:
            messagebox.showinfo("Info", "Nenhuma imagem para salvar.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar imagem",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")]
        )
        if not path:
            return
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(path, bgr):
            self._update_status(f"Imagem salva em: {path}")
        else:
            messagebox.showerror("Erro", "Falha ao salvar imagem.")

    def _get_current_processed_image(self):
        source = None
        if self.video_running:
            with self.lock:
                source = self.video_frame.copy() if self.video_frame is not None else None
        elif self.running:
            with self.lock:
                source = self.frame.copy() if self.frame is not None else None
        else:
            source = self.image_bgr.copy() if self.image_bgr is not None else None

        if source is None:
            return None
        return self.filtros._apply_filters(self, source)

    def _refresh_canvas(self):
        img_rgb = self._get_current_processed_image()
        if img_rgb is not None:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            bgr = self.video_processor.update_tracker(bgr)
            bgr = self.video_processor.detect_purple_bottle(bgr)
            bgr = self.video_processor.detect_template(bgr)
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            h, w = img_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            resized = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            if self.canvas_image_id is None:
                cx, cy = canvas_w // 2, canvas_h // 2
                self.canvas_image_id = self.canvas.create_image(cx, cy, image=img_tk, anchor=tk.CENTER)
            else:
                self.canvas.itemconfig(self.canvas_image_id, image=img_tk)
            self.canvas.image = img_tk
        else:
            self.canvas.delete("all")
            w = self.canvas.winfo_width() or 400
            h = self.canvas.winfo_height() or 300
            self.canvas.create_text(w//2, h//2, text="Carregue uma imagem ou inicie a câmera", fill="#ddd")
            self.canvas_image_id = None

        self.root.after(50, self._refresh_canvas)

    def _update_status(self, text=None):
        if text:
            self.status_var.set(text)
        else:
            self.status_var.set(f"Efeito: {self.effect_var.get()} | Análise: {self.analysis_var.get()}")

    def on_close(self):
        self.stop_camera()
        self.stop_video()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
