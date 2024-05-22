import os
import cv2
import numpy as np

# Create background subtractor
backSub = cv2.createBackgroundSubtractorKNN()

# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Leer las imágenes de las señales de tráfico de una carpeta
folder_path = 'imagenes'
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
signal_names = [os.path.splitext(file)[0] for file in os.listdir(folder_path)]  # Obtener nombres de señales

# Leer y procesar cada imagen de señal de tráfico
reference_images = []
reference_descriptors = []
orb = cv2.ORB_create(nfeatures=1500)  # Incrementar el número de características

for image_path in image_paths:
    reference_image = cv2.imread(image_path)
    if reference_image is None:
        print(f"Error: No se puede leer la imagen '{image_path}'.")
        continue

    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    reference_images.append(reference_image)

    # Detect keypoints and descriptors in the reference image
    kp, des = orb.detectAndCompute(reference_gray, None)
    reference_descriptors.append((kp, des))

# Variable para almacenar la última señal detectada
last_detected_signal = None

while True:
    # Leer un cuadro de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el cuadro de la cámara.")
        break

    # Aplicar substracción de fondo
    fgMask = backSub.apply(frame)

    # Convertir cuadro a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y descriptores en el cuadro actual
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

    if des_frame is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        match_found = False
        
        for i, (kp, reference_des) in enumerate(reference_descriptors):
            matches = bf.match(des_frame, reference_des)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < 50]
            num_good_matches = len(good_matches)
            num_total_matches = len(matches)

            if num_total_matches > 0:
                match_percentage = (num_good_matches / num_total_matches) * 100
                if match_percentage > 50:
                    # Verificar la homografía para asegurar la coincidencia geométrica
                    src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches if m.queryIdx < len(kp_frame)]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches if m.trainIdx < len(kp)]).reshape(-1, 1, 2)

                    if len(src_pts) >= 4 and len(dst_pts) >= 4:  # Necesitamos al menos 4 puntos para calcular la homografía
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        matches_mask = mask.ravel().tolist()

                        if np.sum(matches_mask) > 10:  # Asegurarse de tener suficientes puntos coincidentes
                            match_found = True
                            signal_name = signal_names[i]
                            text = f"{signal_name} - Coincidencia: {match_percentage:.2f}%"
                            print(text)
                            cv2.putText(frame, signal_name, (50, 50 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            # Mostrar la imagen de la señal solo cuando se encuentra una coincidencia
                            cv2.imshow(signal_name, reference_images[i])
                            
                            # Cerrar la ventana de la última señal detectada si es diferente
                            if last_detected_signal is not None and last_detected_signal != signal_name:
                                cv2.destroyWindow(last_detected_signal)
                            last_detected_signal = signal_name
                            break  # If a match is found, no need to check further

        if not match_found:
            # Cerrar la ventana de la última señal detectada si no hay coincidencias
            if last_detected_signal is not None:
                cv2.destroyWindow(last_detected_signal)
                last_detected_signal = None

            cv2.imshow('Matches', frame)
        else:
            matched_img = cv2.drawMatches(frame, kp_frame, reference_images[i], kp, good_matches, None, flags=2)
            cv2.imshow('Matches', matched_img)

    else:
        # Cerrar la ventana de la última señal detectada si no hay descriptores
        if last_detected_signal is not None:
            cv2.destroyWindow(last_detected_signal)
            last_detected_signal = None

        cv2.imshow('Matches', frame)

    # Salir si se presiona Enter (13) o Esc (27)
    keyboard = cv2.waitKey(30)
    if keyboard in (13, 27): 
        break

# Liberar la cámara
cap.release()
cv2.destroyAllWindows()
