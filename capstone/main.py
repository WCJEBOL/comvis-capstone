import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # hitung jumlah objek
    count = len(results[0].boxes)

    # tampilkan angka
    cv2.putText(annotated, f"Count: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()