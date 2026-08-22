import sys

# The single entry point for the whole project:
#   python main.py                 -> opens the GUI
#   python main.py path/to/img.jpg -> runs detection on one image from the
#                                      terminal instead (useful for quick
#                                      tests without clicking through the
#                                      GUI), prints the results, and saves
#                                      annotated images into outputs/


def run_gui():
    from gui.app import GddApp
    GddApp().mainloop()


def run_cli(image_path):
    from pipeline import run_all_detectors, save_annotated_images

    results = run_all_detectors(image_path)
    if results is None:
        print("no_glove_detected")
        sys.exit(1)

    defects = results["defects"]
    holes = defects["holes"]
    fingers = defects["missing_finger"]
    stains = defects["stains"]
    tears = defects["tears"]
    spots = defects["spots"]
    knocks = defects["knocking"]

    print("holes:", "detected" if holes["detected"] else "not detected", holes["count"])
    print("missing_finger:", "detected" if fingers["detected"] else "not detected",
          f"(counted {fingers['finger_count']} fingers)")
    print("stains:", "detected" if stains["detected"] else "not detected", stains["count"])
    print("tears:", "detected" if tears["detected"] else "not detected", tears["count"])
    print("spots:", "detected" if spots["detected"] else "not detected", spots["count"])
    print("knocking:", "detected" if knocks["detected"] else "not detected", knocks["count"])

    save_annotated_images(results)
    print("annotated images saved in the 'outputs' folder")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    else:
        run_gui()
