# import socket
#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(('172.30.1.1', 8585)) # 서버와 연결 시도
#
#
# data = input("send massge : ")
#
# sock.send(data.encode()) # encode를 안해주면 byteSteam이 아니라고 안보내줌
#
#    # data = sock.recv(65535)
#   #  print("돌려받은 데이터 : ", data)
#

import socket
import cv2
import numpy
import time

def openCamera() :

    #OpenCV를 이용해서 webcam으로 부터 이미지 추출
    capture = cv2.VideoCapture(0)

    if (capture.isOpened()):
      print("cap is open")
    else:
      print("cap error")

    return capture



# #연결할 서버(수신단)의 ip주소와 port번호
TCP_IP = '172.30.1.1'
TCP_PORT = 5001

#송신을 위한 socket 준비
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = openCamera()
while True :
    ret, frame = capture.read()

    # 추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    # 추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    print("stringData :", stringData)

    sock.send( str(len(stringData)).ljust(16).encode());
    sock.send( stringData);
    time.sleep(0.1)

    decimg=cv2.imdecode(data,1)
    cv2.imshow('CLIENT',decimg)

    k = cv2.waitKey(1) & 0xff #stop & save
    if (k == ord('s')):
        cv2.imwrite('sample.png', frame)
    elif (k == ord('q')):
        break
    #cv2.destroyAllWindows()

#cv2.imwrite('sample.png', frame)

cv2.destroyAllWindows()
sock.close()