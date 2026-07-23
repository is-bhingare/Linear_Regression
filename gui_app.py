import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np

# Matplotlib Tkinter backend integration
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Import custom Linear Regression Pipeline
from src.linear_regression.model import LinearRegressionPipeline


class StudentRegressionApp(tk.Tk):
    """
    Desktop GUI Application for Student Performance Analysis & Simple Linear Regression.
    Suitable for College Mini-Project Demonstrations.
    """

    def __init__(self):
        super().__init__()

        self.title("Student Performance Analysis Studio — Linear Regression Project")
        self.geometry("1280x820")
        self.minsize(1024, 700)

        # Apply modern Tkinter theme styling
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Color Palette
        self.PRIMARY = "#1e3a8a"      # Navy Blue
        self.ACCENT = "#2563eb"       # Royal Blue
        self.SUCCESS = "#059669"      # Emerald Green
        self.BG_DARK = "#0f172a"      # Dark slate header
        self.BG_LIGHT = "#f8fafc"     # Light background
        self.CARD_BG = "#ffffff"      # White card background
        self.TEXT_DARK = "#1e293b"    # Dark grey text

        self.configure(bg=self.BG_LIGHT)

        # Pipeline Instance
        self.pipeline = LinearRegressionPipeline()
        self.loaded_filepath = None

        # Graph Canvases storage
        self.canvases = {}
        self.toolbars = {}

        # Build UI Components
        self._create_styles()
        self._create_header()
        self._create_main_layout()
        self._create_output_panel()
        self._create_graph_notebook()

        # Load default sample dataset if available
        self._load_default_sample()

    def _create_styles(self):
        """Configure ttk styles."""
        self.style.configure("Header.TFrame", background=self.BG_DARK)
        self.style.configure("HeaderTitle.TLabel", background=self.BG_DARK, foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        self.style.configure("HeaderSub.TLabel", background=self.BG_DARK, foreground="#94a3b8", font=("Segoe UI", 9))

        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=self.ACCENT, foreground="#ffffff", padding=6)
        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background=self.SUCCESS, foreground="#ffffff", padding=6)

        self.style.configure("Card.TLabelframe", background=self.CARD_BG, font=("Segoe UI", 10, "bold"))
        self.style.configure("Card.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground=self.PRIMARY)

    def _create_header(self):
        """Top Header banner."""
        header_frame = ttk.Frame(self, style="Header.TFrame", padding=(15, 12))
        header_frame.pack(fill="x", side="top")

        title = ttk.Label(header_frame, text="🎓 Student Analysis — Simple Linear Regression Studio", style="HeaderTitle.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(header_frame, text="College Mini-Project Demonstration | Interactive Data EDA, Training, Metrics & Prediction Studio", style="HeaderSub.TLabel")
        subtitle.pack(anchor="w")

    def _create_main_layout(self):
        """Main PanedWindow separating Left Controls/Outputs and Right Graph Display."""
        # Top Action Toolbar
        toolbar = ttk.Frame(self, padding=(10, 8), style="Header.TFrame")
        toolbar.pack(fill="x", side="top")

        # Action Buttons
        btn_load = tk.Button(toolbar, text="📂 Load Dataset", command=self.on_load_dataset, bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_load.pack(side="left", padx=4)

        btn_eda = tk.Button(toolbar, text="📊 Show EDA", command=self.on_show_eda, bg="#0d9488", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_eda.pack(side="left", padx=4)

        btn_train = tk.Button(toolbar, text="⚙️ Train Model", command=self.on_train_model, bg="#7c3aed", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_train.pack(side="left", padx=4)

        btn_graphs = tk.Button(toolbar, text="📈 View Graphs", command=self.on_view_graphs, bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_graphs.pack(side="left", padx=4)

        btn_results = tk.Button(toolbar, text="📑 Show Results", command=self.on_show_results, bg="#d97706", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_results.pack(side="left", padx=4)

        btn_predict = tk.Button(toolbar, text="🎯 Predict", command=self.on_predict_popup, bg="#059669", fg="white", font=("Segoe UI", 9, "bold"), padx=12, pady=4, relief="flat", cursor="hand2")
        btn_predict.pack(side="left", padx=4)

        # Feature & Target Selection Frame inside toolbar
        col_frame = ttk.Frame(toolbar, style="Header.TFrame")
        col_frame.pack(side="right", padx=10)

        ttk.Label(col_frame, text="Input (X):", foreground="#e2e8f0", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(5, 2))
        self.combo_feature = ttk.Combobox(col_frame, width=15, state="readonly")
        self.combo_feature.pack(side="left", padx=(0, 10))

        ttk.Label(col_frame, text="Target (y):", foreground="#e2e8f0", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(5, 2))
        self.combo_target = ttk.Combobox(col_frame, width=15, state="readonly")
        self.combo_target.pack(side="left", padx=(0, 5))

        # Main horizontal Split Pane
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Container Frame (Output Console & Prediction Dock)
        self.left_frame = ttk.Frame(self.paned, width=480)
        self.paned.add(self.left_frame, weight=1)

        # Right Container Frame (Graph Notebook + Navigation controls)
        self.right_frame = ttk.Frame(self.paned, width=750)
        self.paned.add(self.right_frame, weight=2)

    def _create_output_panel(self):
        """Create Output Log Panel and Quick Prediction Dock."""
        # Top Label
        output_label_frame = ttk.LabelFrame(self.left_frame, text="📋 Terminal & Model Output Console", style="Card.TLabelframe", padding=8)
        output_label_frame.pack(fill="both", expand=True, side="top", pady=(0, 5))

        # Toolbar above console
        console_tools = ttk.Frame(output_label_frame)
        console_tools.pack(fill="x", side="top", pady=(0, 4))

        btn_clear = tk.Button(console_tools, text="Clear Console", command=self.clear_output, bg="#64748b", fg="white", font=("Segoe UI", 8, "bold"), relief="flat", padx=6, pady=2)
        btn_clear.pack(side="left")

        btn_sample2 = tk.Button(console_tools, text="Load Advertising Data", command=lambda: self.load_csv_file("Advertising.csv"), bg="#3b82f6", fg="white", font=("Segoe UI", 8), relief="flat", padx=6, pady=2)
        btn_sample2.pack(side="right", padx=2)

        # Scrolled Text Box for Output
        self.txt_output = scrolledtext.ScrolledText(
            output_label_frame,
            wrap="word",
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#ffffff",
            padx=8,
            pady=8
        )
        self.txt_output.pack(fill="both", expand=True)

        # Prediction Dock Frame at bottom of left pane
        pred_frame = ttk.LabelFrame(self.left_frame, text="⚡ Quick Interactive Prediction", style="Card.TLabelframe", padding=8)
        pred_frame.pack(fill="x", side="bottom", pady=(5, 0))

        ttk.Label(pred_frame, text="Enter Input Feature Value:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=5)

        self.entry_pred_val = ttk.Entry(pred_frame, width=12, font=("Segoe UI", 10))
        self.entry_pred_val.pack(side="left", padx=5)
        self.entry_pred_val.insert(0, "5.0")

        btn_run_pred = tk.Button(pred_frame, text="Calculate Prediction", command=self.on_run_prediction, bg="#059669", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=8, pady=3, cursor="hand2")
        btn_run_pred.pack(side="left", padx=5)

    def _create_graph_notebook(self):
        """Create Single Window Tabbed Notebook for 3 embedded Matplotlib plots."""
        graph_box = ttk.LabelFrame(self.right_frame, text="📈 Embedded Visualizations (Single Window Tabbed Viewer)", style="Card.TLabelframe", padding=8)
        graph_box.pack(fill="both", expand=True)

        # Notebook Container for Tabs
        self.notebook = ttk.Notebook(graph_box)
        self.notebook.pack(fill="both", expand=True, side="top")

        # Tab Frames
        self.tab_eda = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_test = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_eda, text="  1. EDA Scatter Plot  ")
        self.notebook.add(self.tab_train, text="  2. Training Set Regression  ")
        self.notebook.add(self.tab_test, text="  3. Test Set & Residuals  ")

        # Navigation Controls under Notebook (Slider / Next-Prev buttons)
        nav_control_frame = ttk.Frame(graph_box, padding=4)
        nav_control_frame.pack(fill="x", side="bottom", pady=(5, 0))

        btn_prev = tk.Button(nav_control_frame, text="⏮ Previous Graph", command=self.prev_tab, bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), padx=10, pady=3, relief="flat", cursor="hand2")
        btn_prev.pack(side="left", padx=5)

        self.lbl_tab_indicator = ttk.Label(nav_control_frame, text="Graph 1 of 3: EDA Scatter Plot", font=("Segoe UI", 9, "bold"), foreground=self.PRIMARY)
        self.lbl_tab_indicator.pack(side="left", expand=True)

        btn_next = tk.Button(nav_control_frame, text="Next Graph ⏭", command=self.next_tab, bg="#475569", fg="white", font=("Segoe UI", 9, "bold"), padx=10, pady=3, relief="flat", cursor="hand2")
        btn_next.pack(side="right", padx=5)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def log(self, text: str):
        """Append log message to output text panel."""
        self.txt_output.insert("end", text + "\n")
        self.txt_output.see("end")

    def clear_output(self):
        """Clear console text."""
        self.txt_output.delete("1.0", "end")

    def _load_default_sample(self):
        """Attempt to load default Advertising.csv dataset."""
        if os.path.exists("Advertising.csv"):
            self.load_csv_file("Advertising.csv")
        else:
            self.log("💡 Welcome to Linear Regression Studio!")
            self.log("Click '📂 Load Dataset' to choose a CSV file or select sample data.\n")

    def load_csv_file(self, filepath: str):
        """Load and initialize a CSV dataset."""
        try:
            df = self.pipeline.load_csv(filepath)
            self.loaded_filepath = filepath
            cols = self.pipeline.get_column_names()

            self.combo_feature["values"] = cols
            self.combo_target["values"] = cols

            # Intelligent default selections for Advertising data or Student data
            if "Newspaper" in cols and "Sales" in cols:
                self.combo_feature.set("Newspaper")
                self.combo_target.set("Sales")
            elif "Hours_Studied" in cols and "Exam_Score" in cols:
                self.combo_feature.set("Hours_Studied")
                self.combo_target.set("Exam_Score")
            elif len(cols) >= 2:
                self.combo_feature.set(cols[0])
                self.combo_target.set(cols[-1])

            self.clear_output()
            self.log("============================================================")
            self.log(f"✅ DATASET LOADED SUCCESSFULLY: {os.path.basename(filepath)}")
            self.log("============================================================")
            self.log(f"Path: {os.path.abspath(filepath)}")
            self.log(f"Total Rows: {len(df)} | Total Columns: {len(cols)}")
            self.log(f"Columns: {', '.join(cols)}\n")

            self.log("--- First 5 Rows ---")
            self.log(df.head().to_string())
            self.log("\nReady! Click 'Show EDA' or 'Train Model' to proceed.\n")

        except Exception as e:
            messagebox.showerror("Dataset Load Error", f"Failed to load dataset: {str(e)}")

    def on_load_dataset(self):
        """File open dialog handler."""
        filepath = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            self.load_csv_file(filepath)

    def _validate_columns(self):
        feat = self.combo_feature.get()
        targ = self.combo_target.get()

        if not feat or not targ:
            messagebox.showwarning("Selection Warning", "Please select both Input (X) and Target (y) columns!")
            return None, None

        if feat == targ:
            messagebox.showwarning("Selection Warning", "Input feature (X) and Target (y) cannot be the same column!")
            return None, None

        return feat, targ

    def on_show_eda(self):
        """Action handler: Show EDA statistics and plot."""
        feat, targ = self._validate_columns()
        if not feat or not targ:
            return

        try:
            summary = self.pipeline.get_summary(feat, targ)

            self.log("\n============================================================")
            self.log(f"📊 EXPLORATORY DATA ANALYSIS (EDA): {feat} vs {targ}")
            self.log("============================================================")
            self.log(f"Sample Count: {summary['rows']}")
            self.log(f"Correlation Coefficient: {summary['correlation']:.4f}")

            if summary['correlation'] > 0.7:
                interp = "Strong Positive Linear Relationship ✅"
            elif summary['correlation'] > 0.4:
                interp = "Moderate Positive Relationship 📈"
            else:
                interp = "Weak Linear Relationship ⚠️"
            self.log(f"Relationship Assessment: {interp}\n")

            self.log("--- Summary Statistics ---")
            self.log(summary['describe'].to_string())
            self.log("\n--- Missing Values Check ---")
            self.log(str(summary['null_count']))

            # Render EDA plot
            fig = self.pipeline.generate_eda_figure(feat, targ)
            self._render_graph_in_tab(self.tab_eda, fig, "eda")
            self.notebook.select(self.tab_eda)
            self.log("\n📈 EDA Scatter plot updated in Tab 1.\n")

        except Exception as e:
            messagebox.showerror("EDA Error", f"Failed to generate EDA: {str(e)}")

    def on_train_model(self):
        """Action handler: Train Linear Regression model and log metrics."""
        feat, targ = self._validate_columns()
        if not feat or not targ:
            return

        try:
            metrics = self.pipeline.train(feat, targ)

            self.log("\n============================================================")
            self.log("⚙️ SIMPLE LINEAR REGRESSION MODEL TRAINED")
            self.log("============================================================")
            self.log(f"Input Feature (X): {feat}")
            self.log(f"Target Output (y): {targ}")
            self.log(f"Training Samples: {metrics['train_samples']} | Testing Samples: {metrics['test_samples']}")
            self.log(f"\nModel Equation: {metrics['equation']}")
            self.log(f"Slope (m): {metrics['slope']:.4f}")
            self.log(f"Intercept (c): {metrics['intercept']:.4f}")

            self.log("\n--- Model Evaluation Metrics ---")
            self.log(f"Mean Squared Error (MSE)    : {metrics['mse']:.4f}")
            self.log(f"Root Mean Squared Error(RMSE): {metrics['rmse']:.4f}")
            self.log(f"Mean Absolute Error (MAE)   : {metrics['mae']:.4f}")
            self.log(f"R-squared (R² Score)        : {metrics['r2']:.4f}")

            # Update figures
            fig_eda = self.pipeline.generate_eda_figure()
            fig_train = self.pipeline.generate_train_figure()
            fig_test = self.pipeline.generate_test_figure()

            self._render_graph_in_tab(self.tab_eda, fig_eda, "eda")
            self._render_graph_in_tab(self.tab_train, fig_train, "train")
            self._render_graph_in_tab(self.tab_test, fig_test, "test")

            self.notebook.select(self.tab_train)
            self.log("\n📈 Training & Test Regression Graphs generated in single window!\n")

        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")

    def on_view_graphs(self):
        """Switch view to tabbed graphs."""
        self.notebook.select(self.tab_train)
        self.log("🔎 Switched to Graph Display tab.\n")

    def on_show_results(self):
        """Display actual vs predicted table & mini-project conclusions."""
        if self.pipeline.model is None:
            messagebox.showinfo("Model Not Trained", "Please click 'Train Model' first to generate results!")
            return

        metrics = self.pipeline.metrics
        comp_df = self.pipeline.comparison_df

        self.log("\n============================================================")
        self.log("📑 MODEL EVALUATION RESULTS & PREDICTIONS TABLE")
        self.log("============================================================")
        self.log("--- Test Set: Actual vs Predicted Values ---")
        self.log(comp_df.to_string())

        self.log("\n------------------------------------------------------------")
        self.log("🎓 MINI-PROJECT CONCLUSION & INTERPRETATION")
        self.log("------------------------------------------------------------")
        self.log(f"1. The Simple Linear Regression model learned the relationship: {metrics['equation']}.")
        self.log(f"2. The R² Score of {metrics['r2']:.4f} indicates that {metrics['r2']*100:.1f}% of variance in {self.pipeline.target_col} is explained by {self.pipeline.feature_col}.")
        self.log(f"3. On average, predictions deviate from actual values by RMSE = {metrics['rmse']:.2f}.")
        self.log("4. This confirms that Simple Linear Regression is an effective predictive model for this dataset.\n")

    def on_run_prediction(self):
        """Execute single-value prediction from dock or button."""
        if self.pipeline.model is None:
            messagebox.showwarning("Model Missing", "Please train the model first before making predictions!")
            return

        val_str = self.entry_pred_val.get().strip()
        try:
            input_val = float(val_str)
            pred_val = self.pipeline.predict_single(input_val)

            feat = self.pipeline.feature_col
            targ = self.pipeline.target_col
            m = self.pipeline.metrics['slope']
            c = self.pipeline.metrics['intercept']

            self.log("\n🎯 --- PREDICTION RESULT ---")
            self.log(f"Input ({feat}) = {input_val}")
            self.log(f"Calculation: {targ} = ({m:.4f} * {input_val}) + {c:.4f}")
            self.log(f"Predicted {targ} = {pred_val:.2f}\n")

            messagebox.showinfo("Prediction Result", f"Input ({feat}): {input_val}\nPredicted {targ}: {pred_val:.2f}")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value for prediction!")

    def on_predict_popup(self):
        """Open dedicated Prediction dialog."""
        self.on_run_prediction()

    def _render_graph_in_tab(self, tab: ttk.Frame, fig: plt.Figure, tab_key: str):
        """Embed a Matplotlib Figure into a specified Tkinter Tab."""
        # Clear existing canvas if any
        if tab_key in self.canvases:
            self.canvases[tab_key].get_tk_widget().destroy()
            if tab_key in self.toolbars:
                self.toolbars[tab_key].destroy()

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()

        self.canvases[tab_key] = canvas
        self.toolbars[tab_key] = toolbar

    def prev_tab(self):
        """Navigate to previous graph tab."""
        curr = self.notebook.index(self.notebook.select())
        if curr > 0:
            self.notebook.select(curr - 1)

    def next_tab(self):
        """Navigate to next graph tab."""
        curr = self.notebook.index(self.notebook.select())
        if curr < 2:
            self.notebook.select(curr + 1)

    def _on_tab_changed(self, event):
        """Update tab position label indicator."""
        curr = self.notebook.index(self.notebook.select())
        titles = ["EDA Scatter Plot", "Training Set Regression", "Test Set & Residuals"]
        self.lbl_tab_indicator.config(text=f"Graph {curr + 1} of 3: {titles[curr]}")


if __name__ == "__main__":
    app = StudentRegressionApp()
    app.mainloop()
