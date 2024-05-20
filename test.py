import cv2

def test_webcam():
    # 웹캠 캡처 객체 생성
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

    if not cap.isOpened():
        print("Error: Webcam could not be accessed.")
        return

    # 웹캠이 열리면 프레임을 연속적으로 읽고 표시
    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read video frame.")
            break

        # 프레임을 'Webcam' 창에 표시
        cv2.imshow('Webcam', frame)

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용이 끝나면 캡처 객체와 모든 창을 해제
    cap.release()
    cv2.destroyAllWindows()

# 웹캠 테스트 함수 호출
test_webcam()

