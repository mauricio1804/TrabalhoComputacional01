# interface.py
import tkinter as tk
from tkinter import messagebox, filedialog

class InterfaceBuilder:
    """Responsável por criar menus, botões, canvas e status bar."""

    def __init__(self, app, root):
        """
        :param app: referência para a instância principal (App)
        :param root: janela principal (tk.Tk)
        """
        self.app = app
        self.root = root

    def build_menu(self):
        menubar = tk.Menu(self.root)

        # --- Menu Arquivo ---
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir imagem...", command=self.app.open_image)
        file_menu.add_command(label="Abrir vídeo...", command=self.app.open_video)
        file_menu.add_command(label="Salvar resultado", command=self.app.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.app.on_close)
        menubar.add_cascade(label="Arquivo", menu=file_menu)

        # --- Menu Câmera ---
        camera_menu = tk.Menu(menubar, tearoff=0)
        camera_menu.add_command(label="Iniciar câmera", command=self.app.start_camera)
        camera_menu.add_command(label="Parar câmera", command=self.app.stop_camera)
        menubar.add_cascade(label="Câmera", menu=camera_menu)

        # --- Menu Rastreamento ---
        tracking_menu = tk.Menu(menubar, tearoff=0)
        tracking_menu.add_command(label="Selecionar ROI para rastrear", command=self.app.select_roi)
        tracking_menu.add_command(label="Carregar template...", command=self.app.load_template)
        tracking_menu.add_command(label="Carregar música...", command=self.app.load_music)
        menubar.add_cascade(label="Rastreamento", menu=tracking_menu)

        # --- Menu Ajuda ---
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="Sobre",
            command=lambda: messagebox.showinfo(
                "Sobre",
                "Aplicação de Processamento de Imagens e Vídeo\n"
                "Usando Python + OpenCV + Tkinter\n"
                "Autor: Maurício"
            )
        )
        menubar.add_cascade(label="Ajuda", menu=help_menu)

        self.root.config(menu=menubar)

    def build_controls(self):
        panel = tk.Frame(self.root)
        panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Botões imagem
        tk.Button(panel, text="Abrir imagem", command=self.app.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(panel, text="Abrir vídeo", command=self.app.open_video).pack(side=tk.LEFT, padx=5)
        tk.Button(panel, text="Salvar resultado", command=self.app.save_result).pack(side=tk.LEFT, padx=5)

        # Botões câmera
        tk.Button(panel, text="Iniciar câmera", command=self.app.start_camera).pack(side=tk.LEFT, padx=15)
        tk.Button(panel, text="Parar câmera", command=self.app.stop_camera).pack(side=tk.LEFT, padx=5)

        # Botões vídeo
        tk.Button(panel, text="Iniciar vídeo", command=self.app.start_video).pack(side=tk.LEFT, padx=5)
        tk.Button(panel, text="Parar vídeo", command=self.app.stop_video).pack(side=tk.LEFT, padx=5)

        # Dropdown de efeitos visuais
        effects_frame = tk.LabelFrame(self.root, text="Efeito visual")
        effects_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.app.effect_var = tk.StringVar(value="Nenhum")
        effect_options = [
            "Nenhum", "Cinza", "Negativo", "Otsu",
            "Suavização (Média)", "Suavização (Mediana)",
            "Detector de Bordas (Canny)", "Erosão", "Dilatação",
            "Abertura", "Fechamento"
        ]
        tk.OptionMenu(effects_frame, self.app.effect_var, *effect_options,
                      command=lambda _: self.app._update_status()).pack(side=tk.LEFT, padx=10)

        # Dropdown de análises
        analysis_frame = tk.LabelFrame(self.root, text="Análise de imagem binária")
        analysis_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.app.analysis_var = tk.StringVar(value="Selecione")
        analysis_options = [
            "Selecione", "Histograma (Tonais ou Binário)",
            "Área (pixels brancos)", "Perímetro (contorno)",
            "Diâmetro (máx. distância)", "Contagem de objetos (crescimento de região)"
        ]
        tk.OptionMenu(analysis_frame, self.app.analysis_var, *analysis_options).pack(side=tk.LEFT, padx=10)
        tk.Button(analysis_frame, text="Executar análise", command=self.app.run_analysis).pack(side=tk.LEFT, padx=10)

        # Painel de resultados
        results_frame = tk.LabelFrame(self.root, text="Resultados")
        results_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.app.results_text = tk.Text(results_frame, height=6)
        self.app.results_text.pack(fill=tk.X, padx=6, pady=6)
        self.app.results_text.insert(tk.END, "Selecione uma análise e clique em 'Executar análise'.\n")

    def build_canvas(self):
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.app.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
        self.app.canvas.pack(fill=tk.BOTH, expand=True)
        self.app.canvas_image_id = None

    def build_status(self):
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.app.status_var = tk.StringVar(value="Pronto")
        self.app.status_label = tk.Label(status_frame, textvariable=self.app.status_var, anchor="w")
        self.app.status_label.pack(fill=tk.X)
