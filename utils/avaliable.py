import cv2




cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print(f"웹캠 열 수 없습니다.")
else:
    print(f"웹캠  연결되었습니다.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


