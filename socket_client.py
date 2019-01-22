import socket
import sock_config
import pygame
import xbox360_controller
import time
import lane_detector

# sock = socket.create_connection((sock_config.SERVER_ADDRESS, sock_config.SERVER_PORT))
# sock.sendall("PING".encode('utf-8'))
# sock.close()
#

pygame.init()
# Initialize the joysticks

def right(value):
    center = 110
    if value > 1:
        value = 1
    if value < -1:
        value = -1
    print('VALUE', value)
    return str(center + 30 * value)

def run(value):
    max = 60
    return str(max * (-value))

#
my_controller = xbox360_controller.Controller(0)

def control():
    with socket.create_connection(('192.168.204.105', 10003)) as sock:
        done = False
        while not done:
            # Event processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            left_x, left_y = my_controller.get_left_stick()
            triggers = my_controller.get_triggers()

            wheel_value = "W " + right(left_x) + "\r\n"
            print("Руль: ", wheel_value)
            sock.sendall(wheel_value.encode('utf-8'))

            running = "E " + run(triggers) + "\r\n"
            print("Газ: ", running)
            sock.sendall(running.encode('utf-8'))
            time.sleep(0.1)

def control_test():
    with socket.create_connection(('192.168.204.105', 10003)) as sock:
        done = False
        while not done:
            # Event processing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            left_x, left_y = my_controller.get_left_stick()
            triggers = my_controller.get_triggers()

            video = r'http://192.168.204.105:8081/?action=stream'
            angle = lane_detector.image_pipeline(video)
            for value in angle:

                wheel_value = "W " + right(value / 60) + "\r\n"
                print("Руль: ", wheel_value)
                sock.sendall(wheel_value.encode('utf-8'))

                running = "E " + run(triggers) + "\r\n"
                print("Газ: ", running)
                sock.sendall(running.encode('utf-8'))
                time.sleep(0.1)

if __name__ == "__main__":
    control_test()