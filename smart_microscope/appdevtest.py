import cv2
import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel
from PIL import Image, ImageTk
import os
import time
from pathlib import Path

# GPIO imports (Pi only)
import RPi.GPIO as GPIO

import microfocus
from live_inference import LivePredictor


# ---------------------------------
# Stepper motor sequence
# ---------------------------------
step_sequence = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
]

# ---------------------------------
# GPIO Setup
# ---------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

focus1 = 17
focus2 = 18
focus3 = 27
focus4 = 22

GPIO.setup(focus1, GPIO.OUT)
GPIO.setup(focus2, GPIO.OUT)
GPIO.setup(focus3, GPIO.OUT)
GPIO.setup(focus4, GPIO.OUT)

GPIO.output(focus1, GPIO.LOW)
GPIO.output(focus2, GPIO.LOW)
GPIO.output(focus3, GPIO.LOW)
GPIO.output(focus4, GPIO.LOW)

focus_motor_pins = [focus1, focus2, focus3, focus4]

# Transmission white LED
transIllum = 23
GPIO.setup(transIllum, GPIO.OUT)

stepCount = 100


def switchLED(LEDchannel, on=True):
    GPIO.output(LEDchannel, GPIO.HIGH if on else GPIO.LOW)


def moveMotor(step_count, forward=True, stepPause=0.002):
    direction = False if forward else True
    motor_step_counter = 0

    for _ in range(step_count):
        for pin in range(len(focus_motor_pins)):
            GPIO.output(focus_motor_pins[pin], step_sequence[motor_step_counter][pin])

        if direction:
            motor_step_counter = (motor_step_counter - 1) % 8
        else:
            motor_step_counter = (motor_step_counter + 1) % 8

        time.sleep(stepPause)


def cleanup():
    GPIO.output(focus1, GPIO.LOW)
    GPIO.output(focus2, GPIO.LOW)
    GPIO.output(focus3, GPIO.LOW)
    GPIO.output(focus4, GPIO.LOW)
    GPIO.output(transIllum, GPIO.LOW)
    GPIO.cleanup()


class LoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Connexion Utilisateur")
        self.root.geometry("1920x1080")

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self.create_login_section()

    def create_login_section(self):
        frame = tk.Frame(self.root)
        frame.pack(expand=True)

        font_large = ('Arial', 30)

        tk.Label(frame, text="Connexion", font=('Arial', 35, 'bold')).grid(row=0, columnspan=2, pady=20)

        tk.Label(frame, text="Username:", font=font_large).grid(row=1, column=0, sticky=tk.W, padx=20, pady=10)
        tk.Entry(frame, textvariable=self.username_var, font=font_large).grid(row=1, column=1, padx=20, pady=10)

        tk.Label(frame, text="Password:", font=font_large).grid(row=2, column=0, sticky=tk.W, padx=20, pady=10)
        tk.Entry(frame, textvariable=self.password_var, show="*", font=font_large).grid(row=2, column=1, padx=20, pady=10)

        tk.Button(frame, text="Se connecter", command=self.handle_login, font=font_large).grid(
            row=3, columnspan=2, pady=20
        )

    def handle_login(self):
        username = self.username_var.get()
        password = self.password_var.get()

        if username == "admin" and password == "1234":
            messagebox.showinfo("Connexion réussie", "Bienvenue !")
            self.root.destroy()
            self.open_software_app()
        else:
            messagebox.showerror("Erreur", "Nom d'utilisateur ou mot de passe incorrect.")

    def open_software_app(self):
        app_root = tk.Tk()
        CameraApp(app_root)
        app_root.mainloop()


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application de Caméra")
        self.root.geometry("1920x1080")

        self.cap = None
        self.is_camera_open = False
        self.focus_position = [0, 0]

        self.destPath = tk.StringVar()
        self.imagePath = tk.StringVar()

        # AI settings
        self.model_name = "resnet"   # change here later if you switch models
        self.predictor = LivePredictor(model_name=self.model_name, threshold=0.5)
        self.session_name = time.strftime("slide_%Y%m%d_%H%M%S")
        self.capture_dir = None
        self.csv_path = None
        self.last_prediction_text = ""

        # Video frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(side="left", fill="both", expand=True)

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side="right", fill="y", padx=10, pady=10)

        self.open_camera_btn = tk.Button(button_frame, text="Open Camera", command=self.open_camera)
        self.open_camera_btn.pack(side="top", pady=15, fill="x")

        self.saveLocationEntry = tk.Entry(button_frame, width=25, textvariable=self.destPath)
        self.saveLocationEntry.pack(side="top", pady=15, fill="x")

        self.browseButton = tk.Button(button_frame, width=10, text="BROWSE", command=self.destBrowse)
        self.browseButton.pack(side="top", pady=15, fill="x")

        self.capture_btn = tk.Button(button_frame, text="Capture", command=self.capture_image, state="disabled")
        self.capture_btn.pack(side="top", pady=15, fill="x")

        self.open_image_btn = tk.Button(button_frame, text="Open Image", command=self.open_image)
        self.open_image_btn.pack(side="top", pady=25, fill="x")

        self.autofocus_btn = tk.Button(button_frame, text="Autofocus", command=self.autofocus)
        self.autofocus_btn.pack(side="top", pady=25, fill="x")

        tk.Label(button_frame, text="Manual Focus :").pack(pady=5)

        self.focus_plus_btn = tk.Button(button_frame, text="Focus +", command=lambda: self.manual_focus("plus"))
        self.focus_plus_btn.pack(side="top", pady=15, fill="x")

        self.focus_minus_btn = tk.Button(button_frame, text="Focus -", command=lambda: self.manual_focus("minus"))
        self.focus_minus_btn.pack(side="top", pady=15, fill="x")

        tk.Label(button_frame, text="Manual Scan  :").pack(pady=15)
        arrow_frame = tk.Frame(button_frame)
        arrow_frame.pack(pady=15)

        self.up_btn = tk.Button(arrow_frame, text="▲", command=lambda: self.manual_focus("up"))
        self.up_btn.grid(row=0, column=1)

        self.left_btn = tk.Button(arrow_frame, text="◀", command=lambda: self.manual_focus("left"))
        self.left_btn.grid(row=1, column=0)

        self.down_btn = tk.Button(arrow_frame, text="▼", command=lambda: self.manual_focus("down"))
        self.down_btn.grid(row=1, column=1)

        self.right_btn = tk.Button(arrow_frame, text="▶", command=lambda: self.manual_focus("right"))
        self.right_btn.grid(row=1, column=2)

        self.close_camera_btn = tk.Button(button_frame, text="Close Camera", command=self.close_camera, state="disabled")
        self.close_camera_btn.pack(side="top", pady=15, fill="x")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def ensure_session_paths(self):
        base_dir = self.destPath.get().strip()

        if not base_dir:
            base_dir = str(Path.home() / "pi_tests" / "smart_microscope" / "outputs")

        base_dir = Path(base_dir)
        self.capture_dir = base_dir / self.session_name / "images"
        self.csv_path = base_dir / self.session_name / "live_predictions.csv"

        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra.")
            return

        self.is_camera_open = True
        self.open_camera_btn.config(state="disabled")
        self.capture_btn.config(state="normal")
        self.close_camera_btn.config(state="normal")
        self.led_on()
        self.update_frame()

    def led_on(self):
        switchLED(transIllum, on=True)
        print("LED activée.")

    def led_off(self):
        switchLED(transIllum, on=False)
        print("LED désactivée.")

    def update_frame(self):
        if self.is_camera_open and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.last_prediction_text:
                    cv2.putText(
                        frame,
                        self.last_prediction_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.video_frame.imgtk = img
                self.video_frame.config(image=img)

            self.root.after(10, self.update_frame)

    def destBrowse(self):
        default_dir = str(Path.home() / "pi_tests" / "smart_microscope" / "outputs")
        self.destDirectory = filedialog.askdirectory(initialdir=default_dir)
        if self.destDirectory:
            self.destPath.set(self.destDirectory)

    def capture_image(self):
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Erreur", "Camera is not open.")
            return

        self.ensure_session_paths()

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Erreur", "Impossible de capturer une image.")
            return

        filename = f"capture_{int(time.time() * 1000)}.jpg"
        filepath = self.capture_dir / filename

        cv2.imwrite(str(filepath), frame)

        try:
            result = self.predictor.predict_image(filepath)
            self.predictor.append_to_csv(result, self.csv_path)

            overlay = self.predictor.draw_overlay(frame, result)

            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (1280, 720))
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_frame.imgtk = img
            self.video_frame.config(image=img)

            self.last_prediction_text = f'{result["status"]} | P(M)={result["malignant_probability"]:.2f}'

            messagebox.showinfo(
                "Capture Complete",
                f"Saved: {filepath}\n"
                f"Result: {result['status']}\n"
                f"P(M): {result['malignant_probability']:.3f}\n"
                f"CSV: {self.csv_path}"
            )

        except Exception as e:
            messagebox.showerror("Inference Error", f"Image saved, but prediction failed:\n{e}")

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            top = Toplevel(self.root)
            top.title("Image Ouverte")
            image_label = tk.Label(top)
            image_label.pack()
            img_tk = ImageTk.PhotoImage(img)
            image_label.imgtk = img_tk
            image_label.config(image=img_tk)

    def autofocus(self):
        messagebox.showinfo("Autofocus", "Autofocus placeholder. Real autofocus not yet integrated.")

    def manual_focus(self, direction):
        if direction == "plus":
            moveMotor(stepCount, forward=False)
        elif direction == "minus":
            moveMotor(stepCount, forward=True)
        elif direction == "up":
            self.focus_position[1] -= 1
        elif direction == "down":
            self.focus_position[1] += 1
        elif direction == "left":
            self.focus_position[0] -= 1
        elif direction == "right":
            self.focus_position[0] += 1

    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_camera_open = False
        self.led_off()
        self.video_frame.config(image="")
        self.open_camera_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.close_camera_btn.config(state="disabled")

    def on_close(self):
        try:
            self.close_camera()
        finally:
            cleanup()
            self.root.destroy()

    def __del__(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    LoginApp(root)
    root.mainloop()
