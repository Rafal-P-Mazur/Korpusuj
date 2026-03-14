import os
import sys
import tkinter as tk

splash = None
progress_label = None # Nowa zmienna globalna

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# NOWA FUNKCJA: pozwala innym plikom zmieniać tekst na splashu
def update_status(text):
    global progress_label, splash
    if progress_label and splash:
        progress_label.config(text=text)
        splash.update() # Wymusza odświeżenie okna podczas importu

def show_boot_splash():
    global splash, progress_label
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.configure(bg="#1f2328")

    width = 460
    height = 320
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    splash.geometry(f"{width}x{height}+{x}+{y}")

    frame = tk.Frame(splash, bg="#1f2328", bd=0)
    frame.pack(fill="both", expand=True)

    try:
        from PIL import Image, ImageTk
        logo_path = resource_path("temp/logo.png")
        img = Image.open(logo_path)
        img.thumbnail((250, 120), Image.LANCZOS)
        logo_img = ImageTk.PhotoImage(img)
        splash.logo_img = logo_img
        logo_label = tk.Label(frame, image=logo_img, bg="#1f2328", bd=0)
        logo_label.pack(pady=(20, 0))
        title_pady = (10, 5)
    except Exception:
        title_pady = (40, 10)

    tk.Label(frame, text="Korpusuj", font=("Verdana", 22, "bold"), fg="white", bg="#1f2328").pack(pady=title_pady)
    tk.Label(frame, text="Uruchamianie aplikacji, proszę czekać...", font=("Verdana", 11), fg="#cfd8dc", bg="#1f2328").pack(pady=(0, 20))

    # Zapisujemy referencję do labela, żeby móc go zmieniać
    progress_label = tk.Label(frame, text="Inicjalizacja...", font=("Verdana", 10), fg="#9fb3c8", bg="#1f2328")
    progress_label.pack()

    splash.update_idletasks()
    splash.update()

def close_splash():
    global splash
    if splash:
        try:
            splash.destroy()
        except Exception:
            pass
        splash = None

def main():
    show_boot_splash()
    import engine # Podczas tego importu engine.py będzie wywoływał update_status
    close_splash()
    if hasattr(engine, "main"):
        engine.main()

if __name__ == "__main__":
    main()