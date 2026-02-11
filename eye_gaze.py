import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def eye_direction(eye_points):
    x_coords = [p[0] for p in eye_points]
    avg_x = np.mean(x_coords)

    # Thresholds (tuned manually)
    if avg_x < 0.42:
        return "Looking Left"
    elif avg_x > 0.58:
        return "Looking Right"
    else:
        return "Looking Center"

def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for lm in result.multi_face_landmarks:
                    left_eye_pts = []
                    right_eye_pts = []

                    for idx in LEFT_EYE:
                        x = int(lm.landmark[idx].x * w)
                        y = int(lm.landmark[idx].y * h)
                        left_eye_pts.append((lm.landmark[idx].x, lm.landmark[idx].y))
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    for idx in RIGHT_EYE:
                        x = int(lm.landmark[idx].x * w)
                        y = int(lm.landmark[idx].y * h)
                        right_eye_pts.append((lm.landmark[idx].x, lm.landmark[idx].y))
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                    # Calculate directions
                    left_dir = eye_direction(left_eye_pts)
                    right_dir = eye_direction(right_eye_pts)

                    # Combine both eyes
                    if left_dir == right_dir:
                        direction = left_dir
                    else:
                        direction = "Uncertain"

                    cv2.putText(frame, direction, (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Eye Gaze Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
