import os
import sys
import tempfile

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# This file lives inside the gui/ folder, but it needs to import main.py and
# the preprocessing/detection/defects packages that live one level up, at
# the project root. This adds the project root to Python's search path so
# those imports work no matter where this script is run from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import run_all_detectors
from preprocessing.image_prep import resize_max_dim

# The order the six defects are shown in the results list, and the friendly
# text shown for each one.
DEFECT_LABELS = {
    "holes": "Holes",
    "missing_finger": "Missing Finger",
    "stains": "Stains",
    "tears": "Tears",
    "spots": "Spots",
    "knocking": "Knocking",
}

# Tkinter's PhotoImage widget can only load images from files on disk (or
# from raw image bytes), it cannot show an OpenCV image (a NumPy array)
# directly. So every time we want to display an image, we first save it as
# a PNG file into this temporary folder, then load that PNG file into the
# GUI. The folder is created once when the program starts.
TEMP_FOLDER = tempfile.mkdtemp(prefix="gdd_gui_")
PREVIEW_IMAGE_PATH = os.path.join(TEMP_FOLDER, "preview.png")

DISPLAY_MAX_DIM = 500  # shrink big photos so they fit nicely in the window


class GddApp(tk.Tk):
    """Main window of the Gloves Defect Detection GUI."""

    def __init__(self):
        super().__init__()

        self.title("Gloves Defect Detection (GDD)")
        self.geometry("900x650")

        # State kept between button clicks.
        self.selected_image_path = None   # path to the photo the user picked
        self.detection_results = None     # dict returned by run_all_detectors
        self.current_photo_image = None   # keeps a reference so Tkinter does
                                           # not garbage-collect the displayed
                                           # image (a common Tkinter gotcha)

        self._build_widgets()

    def _build_widgets(self):
        # --- top row: Browse button, Run Detection button, selected file name ---
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        browse_button = tk.Button(top_frame, text="Browse", command=self.on_browse_clicked)
        browse_button.pack(side=tk.LEFT)

        run_button = tk.Button(top_frame, text="Run Detection", command=self.on_run_detection_clicked)
        run_button.pack(side=tk.LEFT, padx=10)

        self.selected_file_label = tk.Label(top_frame, text="No image selected")
        self.selected_file_label.pack(side=tk.LEFT, padx=10)

        # --- middle row: image on the left, defect list on the right ---
        middle_frame = tk.Frame(self)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(middle_frame, text="(image will appear here)",
                                     relief=tk.SUNKEN)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(middle_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        results_title = tk.Label(right_frame, text="Defect results (click one to view it):")
        results_title.pack(side=tk.TOP, anchor="w")

        self.results_listbox = tk.Listbox(right_frame, width=40, height=15)
        self.results_listbox.pack(side=tk.TOP, fill=tk.Y)
        self.results_listbox.bind("<<ListboxSelect>>", self.on_defect_selected)

        # --- bottom row: status messages, e.g. "No glove detected" ---
        self.status_label = tk.Label(self, text="", fg="red")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

    def on_browse_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Choose a glove photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not file_path:
            return  # user closed the dialog without picking a file

        self.selected_image_path = file_path
        self.selected_file_label.config(text=os.path.basename(file_path))
        self.status_label.config(text="")

        # Clear any results from a previous image.
        self.detection_results = None
        self.results_listbox.delete(0, tk.END)

        # Show the original photo the user picked, before running detection.
        original_image = cv2.imread(file_path)
        if original_image is not None:
            self._show_image(original_image)

    def on_run_detection_clicked(self):
        if self.selected_image_path is None:
            messagebox.showwarning("No image", "Please choose an image first (Browse button).")
            return

        try:
            # This calls the exact same pipeline as main.py:
            # load image -> preprocess -> detect_glove -> run all 6 detectors.
            self.detection_results = run_all_detectors(self.selected_image_path)
        except Exception as error:
            messagebox.showerror("Error", f"Could not process this image:\n{error}")
            return

        if self.detection_results is None:
            self.status_label.config(text="No glove detected")
            self.results_listbox.delete(0, tk.END)
            return

        self.status_label.config(text="")
        self._fill_results_list()

        # Automatically show the first defect's annotated image, so the user
        # sees a result right away without needing to click the list first.
        first_defect_name = list(self.detection_results.keys())[0]
        self.results_listbox.selection_set(0)
        self._show_defect_image(first_defect_name)

    def _fill_results_list(self):
        self.results_listbox.delete(0, tk.END)
        for defect_name, result in self.detection_results.items():
            label = DEFECT_LABELS[defect_name]
            status = "detected" if result["detected"] else "not detected"
            count = result["count"]
            row_text = f"{label}: {status} (count: {count})"
            self.results_listbox.insert(tk.END, row_text)

    def on_defect_selected(self, event):
        if self.detection_results is None:
            return

        selected_indexes = self.results_listbox.curselection()
        if not selected_indexes:
            return

        selected_index = selected_indexes[0]
        defect_name = list(self.detection_results.keys())[selected_index]
        self._show_defect_image(defect_name)

    def _show_defect_image(self, defect_name):
        result = self.detection_results[defect_name]
        annotated_image = result["annotated_image"]
        self._show_image(annotated_image)

    def _show_image(self, bgr_image):
        """Display an OpenCV image (a BGR NumPy array) in the image_label."""
        # Shrink very large photos so they fit inside the window.
        resized_image = resize_max_dim(bgr_image, max_dim=DISPLAY_MAX_DIM)

        # Tkinter cannot display a NumPy array directly, so we save it to a
        # PNG file on disk first, then load that file into a PhotoImage.
        cv2.imwrite(PREVIEW_IMAGE_PATH, resized_image)
        photo_image = tk.PhotoImage(file=PREVIEW_IMAGE_PATH)

        self.image_label.config(image=photo_image, text="")
        self.current_photo_image = photo_image  # keep a reference alive


if __name__ == "__main__":
    app = GddApp()
    app.mainloop()
