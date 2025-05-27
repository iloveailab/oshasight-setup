# OshaSight – live webcam monitor for PPE & smoke

import os
import cv2
import time
import subprocess
import PySimpleGUI as sg
from ultralytics import YOLO

# we define thresholds for classes, 
# because precision varies due to the results of training datasets
thresholds = {
    'Mask': 0.85, 
    'Smoke': 0.85, 
    'Gloves': 0.60,
    'Helmet': 0.50, 
    'Goggles': 0.80, 
    'Vest': 0.50
}

# different borders for different classes (frame border)
borders = {
    'Mask': (0, 255, 0), # green
    'Gloves': (255, 255, 0), # yellow
    'Smoke': (255, 0, 0), # blue
    'Helmet': (0, 0, 255), # red
    'Goggles': (0, 255, 255), # cyan
    'Vest': (255, 0, 255) # magenta
}

# we put explaination for classes, so that user can understand what classes mean
expl = {
    'Mask': 'Bad air',
    'Smoke': 'No-smoking zone',
    'Gloves': 'Hand protection',
    'Helmet': 'Head protection',
    'Goggles': 'Eye protection',
    'Vest': 'Hi-viz req'
}

# we load yolo models from runs folder
# we reach to best.pt files for each class
models = {}
models['Mask'] = YOLO('runs/mask/weights/best.pt')
models['Smoke'] = YOLO('runs/smoke/weights/best.pt')
models['Gloves'] = YOLO('runs/gloves/weights/best.pt')
models['Helmet'] = YOLO('runs/helmet/weights/best.pt')
models['Goggles'] = YOLO('runs/goggles/weights/best.pt')
models['Vest'] = YOLO('runs/vest/weights/best.pt')

# audio for smoke alarm
audio_file = 'caution.wav'

# for our classes
checkboxes = []
for cls in models:
    checkbox_key = cls + '_chk'  # simpler key naming
    checkboxes.append([sg.Checkbox(cls, key=checkbox_key, default=True)])

# for the panel
table_head = ['ID', 'Class', 'expl']
rows = [[i + 1, cls, expl[cls]] for i, cls in enumerate(models)]

# the general panel structure
layout = [
    [sg.Text('OshaSight - Live Safety Monitor', font=('Helvetica', 16, 'bold'))],
    [sg.Image('', key='img'),
     sg.Column([
         [sg.Frame('Toggle Classes', checkboxes)],
         [sg.Table(rows,
                   headings=table_head,
                   justification='left',
                   col_widths=[4, 10, 22],
                   num_rows=len(models))]
     ])]
]

window = sg.Window('OshaSight', layout, location=(100, 100), finalize=True)


# smoke alarm audio <-> function
def play_audio_once():
    subprocess.Popen(
        ['powershell', '-c',
         f"(New-Object Media.SoundPlayer '{audio_file}').PlaySync();"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

last_smoke_time = 0
alerting = False

webcam = cv2.VideoCapture(0)

while True:
    event, _ = window.read(timeout=1)

    _, frame = webcam.read()

    now = time.time()
    smoke_detected = False

    for cls, model in models.items():
        if not window[cls + '_chk'].get():
            continue

        conf_th = thresholds[cls]
        preds = model.predict(frame, conf=conf_th)[0].boxes
        for b in preds:
            if float(b.conf[0]) < conf_th:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), borders[cls], 2)
            cv2.putText(frame, f'{cls} {b.conf[0]:.2f}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, borders[cls], 1)
            if cls == 'Smoke':
                smoke_detected = True

    # smoke detection logic
    if smoke_detected:
        print("Smoke detected.")
        if not alerting and now - last_smoke_time > 3:
            play_audio_once()
            alerting = True
        last_smoke_time = now
    
    # alert logic
    if alerting:
        if now - last_smoke_time > 5:
            alerting = False
            print("Smoke alert ended.")
        else:
            overlay = frame.copy()
            overlay[:] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.putText(frame, '⚠  NO SMOKING', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    window['img'].update(data=cv2.imencode('.png', frame)[1].tobytes())

webcam.release()
window.close()
