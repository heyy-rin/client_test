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

# return : camera 객체
# 카메라가 정상적으로 작동하는지 확인함
def openCamera() :

    #OpenCV를 이용해서 webcam으로 부터 이미지 추출
    capture = cv2.VideoCapture(0)

    if (capture.isOpened()):
      print("cap is open")
    else:
      print("cap error")

    return capture

# param : frame 보내려는 이미지
# return : 이미지를 배열로 만든 결과
# frame을 이미지 배열로 만들고 인코딩한다
# 이미지의 크기와 이미지 배열을 분리해서 전송한다
# 이미지 전송 후 전송된 이미지 배열을 반환
def sendImage(frame) :

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

    # 보내는 데이터 확인
    print("stringData :", stringData)

    # 데이터 전송
    sock.send( str(len(stringData)).ljust(16).encode()); # 이미지 크기 먼저 저송
    sock.send( stringData );    # 이미지 배열 전송
    time.sleep(0.1) # 전송속도가 너무 빠르면 안됨!

    return data;


#연결할 서버(수신단)의 ip주소와 port번호
#TCP_IP = '172.30.1.1'
TCP_IP = '10.10.24.117'

TCP_PORT = 5001

#송신을 위한 socket 준비
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = openCamera()
while True :
    ret, frame = capture.read()

    #


    #

    # 인코딩하기 전에 사람 검출한 뒤 ! 해당되는 이미지만 보내기!

    # 전송된 이미지가 원하는 결과였는지 확인하기 위해 client에서 출력
    data = sendImage(frame) # 전송된 이미지배열이 저장됨!
    decimg=cv2.imdecode(data,1) # 이미지 배열을 decode
    cv2.imshow('CLIENT',decimg) # decode된 이미지를 확인해서 원하는 이미지가 전송되었는지 확인

    # cv2가 저장 또는 중지를 위해 키보드 입력을 기다림
    k = cv2.waitKey(1) & 0xff # stop & save
    if (k == ord('s')):
        cv2.imwrite('sample.png', frame) # save기능은 단순 확인용
    elif (k == ord('q')):
        break
    #cv2.destroyAllWindows()

#cv2.imwrite('sample.png', frame)

cv2.destroyAllWindows()
sock.close()