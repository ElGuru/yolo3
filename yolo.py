from ultralytics import YOLO
import cv2
import math

# Carga del modelo YOLO
model = YOLO(r"C:\Users\SISTEMAS\Desktop\Codes\python\IA\Lab_09\yolo11n.pt")

# Lista de clases deseadas
classNames = ["bicycle", "car", "motorbike", "bus", "truck", "boat",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Configura la captura de video
captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, img = captura.read()
    if not success:
        break

    # Inferencia del modelo
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Mostrar solo si la clase está en la lista
            if class_name in classNames:
                # Coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]), 2)

                # Dibujar caja y texto
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # También puedes imprimir por consola
                print(f'Detected: {class_name} (Confidence: {confidence})')

    # Mostrar resultado en ventana
    cv2.imshow('Webcam', img)

    # Salir con 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()
