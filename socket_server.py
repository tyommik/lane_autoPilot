import socket

#

import RPi.GPIO as GPIO
import time

class CarServoControl():
    def __init__(self):
        # Параметры для управления сервоприводом
        SERVO_PIN = 24

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(24, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(24, 100)
        self.servo_pwm.start(5)
        self.servo_pwm.ChangeDutyCycle(13.5)

    def update(self, angle):
        duty = float(angle) / 10.0 + 2.5
        print("Servo control: {duty}".format(duty=duty))
        self.servo_pwm.ChangeDutyCycle(duty)

class CarEngineControl():
    def __init__(self):
        PIN = 18
        self.PWMA1 = 6
        self.PWMA2 = 13
        self.PWMB1 = 20
        self.PWMB2 = 21

        D1 = 12
        D2 = 26

        PWM = 30

        #SERVO_PIN = 24

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PIN, GPIO.IN, GPIO.PUD_UP)
        #GPIO.setup(SERVO_PIN, GPIO.OUT)
        GPIO.setup(self.PWMA1, GPIO.OUT)
        GPIO.setup(self.PWMA2, GPIO.OUT)
        GPIO.setup(self.PWMB1, GPIO.OUT)
        GPIO.setup(self.PWMB2, GPIO.OUT)
        GPIO.setup(D1, GPIO.OUT)
        GPIO.setup(D2, GPIO.OUT)
        self.p1 = GPIO.PWM(D1, 500) # M1
        self.p2 = GPIO.PWM(D2, 500) # M2
        self.p1.start(0)
        self.p2.start(0)

        GPIO.output(self.PWMA1, 1)
        GPIO.output(self.PWMA2, 0)
        self.p1.ChangeDutyCycle(20)

    def moving_forward(self):
        GPIO.output(self.PWMB1, 1)
        GPIO.output(self.PWMB2, 0)

    def moving_backward(self):
        GPIO.output(self.PWMB1, 0)
        GPIO.output(self.PWMB2, 1)

    def right(self):
        GPIO.output(self.PWMA1, 1)
        GPIO.output(self.PWMA2, 0)

    def left(self):
        GPIO.output(self.PWMA1, 0)
        GPIO.output(self.PWMA2, 1)

    def update(self, angle):
        if int(angle) > 25:
            self.moving_forward()
            duty = int(angle)
            print(duty)
            self.p2.ChangeDutyCycle(duty)
        elif int(angle) < 0:
            self.moving_backward()
            duty = int(angle)
            print(duty)
            self.p2.ChangeDutyCycle(abs(duty))
        else:
            self.p2.ChangeDutyCycle(0)

class CarControl():
    """ Класс управления автомобилем """
    def __init__(self, address = '', port=10001):
        self.address = address
        self.port = port

        # Параметры для управления сервоприводом
        self.servo = CarServoControl()
        self.eng = CarEngineControl()

    def wheel(self, angle):
        self.servo.update(angle)
    def engine(self, angle):
        self.eng.update(-angle)

def _read(conn):
    data = b''
    while not data.endswith(b"\n\n"):
        try:
            data += conn.recv(2048)
        except socket.error as err:
            print(err)
            break

if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("", 10000))
        sock.listen(socket.SOMAXCONN)

        while True:
            conn, addr = sock.accept()
            conn.settimeout(100)
            client_address, client_port = addr
            print('Подключился клиент с адресом {client_address}, порт {client_port}'.format(client_address=client_address, client_port=client_port))
            car = CarControl()
            with conn:

                while True:
                    data = b""
                    while not data.endswith(b'\r\n'):
                        try:
                            data += conn.recv(2048)
                        except socket.timeout as err:
                            print('Timeout')
                            break
                    if not data:
                        break
                    try:
                        if data.decode('utf-8').startswith('W'):
                            _, angle = data.decode('utf-8').split(' ')
                            car.wheel(float(angle))
                        elif data.decode('utf-8').startswith('E'):
                            _, angle = data.decode('utf-8').split(' ')
                            car.engine(float(angle))
                        else:
                            print('Wrong command')
                    except ValueError:
                        pass

