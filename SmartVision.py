import cv2
import threading
import time
import numpy as np
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


# ============== CAMERA PROCESSOR ==============
class CameraProcessor:
    # YOLOv8 class IDs we care about
    PERSON_ID = 0
    FACE_IDS = {0}  # fallback: treat person as face if no face model

    def __init__(self):
        print("[INFO] Loading YOLOv8n model...")
        self.model = YOLO("yolov8n.pt")  # auto-downloads on first run
        print("[OK] Model loaded")

        print("[INFO] Opening camera...")
        self.cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None and test.mean() > 1:
                    self.cap = cap
                    print(f"[OK] Camera opened (backend {backend})")
                    break
                cap.release()

        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = False
        self.current_frame = None
        self.lock = threading.Lock()

        self.fps = 0.0
        self.person_count = 0
        self.fps_buffer = deque(maxlen=20)
        self.last_time = time.time()

        # settings
        self.conf_thresh = 0.45
        self.iou_thresh = 0.4

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)

    def release(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def _process(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Run YOLO inference — imgsz=320 is fastest on CPU
            results = self.model(
                frame,
                imgsz=320,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=[0],        # only 'person' class
                verbose=False
            )

            person_count = 0
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == 0:  # person
                        person_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        label = f"person {conf:.0%}"
                        cv2.putText(frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)

            self.person_count = person_count

            # FPS
            now = time.time()
            dt = now - self.last_time
            if dt > 0:
                self.fps_buffer.append(1.0 / dt)
                self.fps = sum(self.fps_buffer) / len(self.fps_buffer)
            self.last_time = now

            with self.lock:
                self.current_frame = frame


# ============== UI ==============
class DetectorUI:
    DARK  = '#0f0f0f'
    PANEL = '#1a1a1a'
    CARD  = '#232323'
    ACC   = '#00e676'
    ORG   = '#ffd740'
    RED   = '#ff1744'
    TEXT  = '#e0e0e0'
    MUTED = '#555555'

    def __init__(self, processor: CameraProcessor):
        self.processor = processor
        self.root = tk.Tk()
        self.root.title("Vision Detector")
        self.root.geometry("1080x640")
        self.root.configure(bg=self.DARK)
        self.root.resizable(True, True)
        self._build()
        self._tick()

    def _build(self):
        # ── left: video ──────────────────────────────────
        left = tk.Frame(self.root, bg=self.DARK)
        left.pack(side='left', fill='both', expand=True, padx=(12, 6), pady=12)

        hdr = tk.Frame(left, bg=self.DARK)
        hdr.pack(fill='x', pady=(0, 8))
        tk.Label(hdr, text="VISION DETECTOR",
                 font=('Courier', 13, 'bold'), bg=self.DARK, fg=self.ACC).pack(side='left')
        self.status_lbl = tk.Label(hdr, text="● IDLE",
                                   font=('Courier', 10), bg=self.DARK, fg=self.MUTED)
        self.status_lbl.pack(side='right')

        wrap = tk.Frame(left, bg='#111', highlightbackground='#2a2a2a', highlightthickness=1)
        wrap.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(wrap, bg='#111', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # ── right: panel ─────────────────────────────────
        right = tk.Frame(self.root, bg=self.PANEL, width=240)
        right.pack(side='right', fill='y', padx=(0, 12), pady=12)
        right.pack_propagate(False)

        # stats
        tk.Label(right, text="STATS", font=('Courier', 9),
                 bg=self.PANEL, fg=self.MUTED).pack(anchor='w', padx=16, pady=(20, 4))

        for attr, label, color in [
            ('fps_var',    'FPS',    self.ORG),
            ('people_var', 'PEOPLE', self.ACC),
        ]:
            card = tk.Frame(right, bg=self.CARD, padx=14, pady=10)
            card.pack(fill='x', padx=14, pady=(0, 6))
            tk.Label(card, text=label, font=('Courier', 9),
                     bg=self.CARD, fg=self.MUTED).pack(anchor='w')
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            tk.Label(card, textvariable=var,
                     font=('Courier', 26, 'bold'), bg=self.CARD, fg=color).pack(anchor='w')

        # settings
        tk.Label(right, text="SETTINGS", font=('Courier', 9),
                 bg=self.PANEL, fg=self.MUTED).pack(anchor='w', padx=16, pady=(16, 4))

        self._slider(right, "Confidence", 0.1, 0.9, self.processor.conf_thresh,
                     lambda v: setattr(self.processor, 'conf_thresh', float(v)))
        self._slider(right, "IOU Thresh", 0.1, 0.9, self.processor.iou_thresh,
                     lambda v: setattr(self.processor, 'iou_thresh', float(v)))

        # buttons
        bf = tk.Frame(right, bg=self.PANEL)
        bf.pack(fill='x', padx=14, pady=16)

        self.start_btn = tk.Button(bf, text="START", command=self._start,
                                   bg=self.ACC, fg='#000', font=('Courier', 10, 'bold'),
                                   relief='flat', cursor='hand2', pady=6)
        self.start_btn.pack(fill='x', pady=(0, 6))

        self.stop_btn = tk.Button(bf, text="STOP", command=self._stop,
                                  bg='#333', fg=self.MUTED, font=('Courier', 10, 'bold'),
                                  relief='flat', cursor='hand2', pady=6, state='disabled')
        self.stop_btn.pack(fill='x', pady=(0, 6))

        tk.Button(bf, text="QUIT", command=self._quit,
                  bg='#222', fg=self.MUTED, font=('Courier', 10),
                  relief='flat', cursor='hand2', pady=6).pack(fill='x')

    def _slider(self, parent, label, from_, to, initial, cb):
        wrap = tk.Frame(parent, bg=self.CARD, padx=14, pady=8)
        wrap.pack(fill='x', padx=14, pady=(0, 6))
        top = tk.Frame(wrap, bg=self.CARD)
        top.pack(fill='x')
        tk.Label(top, text=label, font=('Courier', 8),
                 bg=self.CARD, fg=self.MUTED).pack(side='left')
        val_lbl = tk.Label(top, text=f"{initial:.2f}",
                           font=('Courier', 8, 'bold'), bg=self.CARD, fg=self.ACC)
        val_lbl.pack(side='right')
        s = tk.Scale(wrap, from_=from_, to=to, orient='horizontal', resolution=0.01,
                     bg=self.CARD, fg=self.TEXT, highlightthickness=0,
                     troughcolor='#333', activebackground=self.ACC, showvalue=False, bd=0)
        s.set(initial)
        s.pack(fill='x', pady=(4, 0))
        s.config(command=lambda v: (val_lbl.config(text=f"{float(v):.2f}"), cb(v)))

    def _start(self):
        self.processor.start()
        self.start_btn.config(state='disabled', bg='#333', fg=self.MUTED)
        self.stop_btn.config(state='normal', bg=self.RED, fg='#fff')
        self.status_lbl.config(text="● RUNNING", fg=self.ACC)
        self.fps_var.set("0")
        self.people_var.set("0")

    def _stop(self):
        self.processor.stop()
        self.start_btn.config(state='normal', bg=self.ACC, fg='#000')
        self.stop_btn.config(state='disabled', bg='#333', fg=self.MUTED)
        self.status_lbl.config(text="● IDLE", fg=self.MUTED)
        self.fps_var.set("—")

    def _quit(self):
        self.processor.stop()
        self.processor.release()
        self.root.quit()
        self.root.destroy()

    def _tick(self):
        frame = self.processor.get_frame()
        if frame is not None:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                h, w = frame.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tk_img = ImageTk.PhotoImage(pil_img)
                self.canvas.delete("all")
                self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                                         image=tk_img, anchor='nw')
                self.canvas.image = tk_img

        if self.processor.running:
            self.fps_var.set(f"{self.processor.fps:.1f}")
            self.people_var.set(str(self.processor.person_count))

        self.root.after(33, self._tick)

    def run(self):
        self.root.mainloop()


# ============== MAIN ==============
def main():
    print("=" * 50)
    print("   VISION DETECTOR  —  YOLOv8n")
    print("=" * 50)
    try:
        processor = CameraProcessor()
        DetectorUI(processor).run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        input("Enter to exit...")

if __name__ == '__main__':
    main()