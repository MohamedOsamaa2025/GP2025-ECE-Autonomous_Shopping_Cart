# final!  new_ip_0.13, new_model_handOcc

import cv2
import torch
from ultralytics import YOLO
from collections import deque
import numpy as np
import asyncio
import websockets
import nest_asyncio
import os
import paho.mqtt.client as mqtt
import time
import json
import ssl

with open("small_labels.json", "r") as file:
    products = json.load(file)

broker_url = '192.168.0.13'
broker_port = 1883
topic = "flutter/pub"

# # Initialize MQTT client globally
mqtt_client = mqtt.Client()
mqtt_client.connect(broker_url, broker_port)

added = 0
removed = 0
count = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(r"best (6) handOcc.pt").to(device)
nest_asyncio.apply()####
conf_threshold = 0.5

async def infere(frame):
    global conf_threshold
    global added, removed, count

    counted_ids = set()
    track_duration = {}  # Track how long each ID has been in view
    max_duration = 0#*INC from 3->8*## Minimum frames before acting on a track_id
    bbox_history = deque(maxlen=1)#*DEC from 5->1*#
    no_detection_counter = 0

    # frame = cv2.resize(frame, (480, 480))  # or smaller if acceptable
    np_arr = np.frombuffer(frame, np.uint8)####
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)####

    frame_height = frame.shape[0]
    lower_bound = int(0.30 * frame_height)
    higher_bound = lower_bound
    # higher_bound = frame_height - lower_bound

    display_frame = frame.copy()

    results = model.track(frame, conf=conf_threshold, persist=True)
    detections = []

    # if None in results: #checks if the results list itself contains None (never happens, it always contain at least shape attr)
    if not results[0].obb:  # Correct way to check for no detections (empty frame)   
        no_detection_counter += 1
        if no_detection_counter >= 50:     #100frames ~= 3-5sec => use50-100frames
            bbox_history.clear()
            no_detection_counter = 0
        # continue

    # rslts = [instance1, instance2, ...] 
    for result in results:      #rslt->instance(obb, speed, shap) 
        for obb in result.obb:  #obb->instance (xyxyxyxy, conf, cls, id)
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, obb.xyxyxyxy[0].cpu().numpy().flatten())
            conf = float(obb.conf.cpu().numpy())
            cls = int(obb.cls.cpu().numpy())
            track_id = int(obb.id.cpu().numpy()) if obb.id is not None else None

            if conf >= conf_threshold:
                ######### prevent fluctuations, not empty frames #########
                if None in [x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, track_id]:
                    # detections.append([0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0])
                    print("hi")
                else:
                    detections.append([x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, track_id])

            else:
                detections.append([0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0])

    if detections:
        detections.sort(key=lambda x: x[8], reverse=True)
        bbox_history.append(detections[0])  # Smooth only top detection
        valid_boxes = [b for b in bbox_history if b is not None]    # ensure all old boxs valid (init boxs)
        if len(valid_boxes) == bbox_history.maxlen:
            avg_box = np.mean(valid_boxes, axis=0).astype(int)
            detections = [avg_box]
        else:
            detections = []


        current_ids = {det[-1] for det in detections if det[-1] is not None}
        for tid in current_ids:
            track_duration[tid] = track_duration.get(tid, 0) + 1

        track_duration = {k: v for k, v in track_duration.items() if k in current_ids}
        counted_ids.intersection_update(current_ids)
        # fileprint("second")
        for x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, track_id in detections:
            # fileprint("first")
            draw(x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, model, display_frame)

            center_y = int((y1 + y2 + y3 + y4) / 4)
            crossUpB = check_crossing_directionU(track_id, center_y, lower_bound)
            crossDnB = check_crossing_directionD(track_id, center_y, higher_bound)
            
            if crossUpB == 'up' and track_duration.get(track_id, 0) > max_duration:
                count -= 1
                remove_from_cart(str(cls))
            elif crossDnB == 'down' and track_duration.get(track_id, 0) > max_duration:
                count += 1
                add_to_cart(str(cls))

    # Draw crossing boundaries
    cv2.line(display_frame, (0, higher_bound), (frame.shape[1], higher_bound), (255, 0, 0), 2)
    cv2.line(display_frame, (0, lower_bound), (frame.shape[1], lower_bound), (255, 0, 0), 2)

    # Draw count label
    label = f"#count: {count}"
    cv2.putText(display_frame, label, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 9)
    cv2.putText(display_frame, label, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    track_id_label = f"track_id: {track_id if 'track_id' in locals() and track_id is not None else 'None'}"
    cv2.putText(display_frame, track_id_label, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 9)
    cv2.putText(display_frame, track_id_label, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    print(track_id_label)

    cv2.imshow("YOLOv8 OBB Inference", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def fileprint(x):
    with open("log.txt", "a") as file:
        file.write(x)
        file.write("\n")
#**remove**
def draw(x1, y1, x2, y2, x3, y3, x4, y4, conf, cls, model, frame):
    label = f"{model.names[int(cls)]} {conf:.2f}"
    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
    
    # Find the REAL top-left corner (minimum x + y value)
    top_left = corners[np.argmin(corners.sum(axis=1))]
    cv2.polylines(frame, [corners.reshape((-1, 1, 2))], 
                 isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Calculate text position (offset slightly inside the box)
    text_x = top_left[0] + 5
    text_y = top_left[1] + 20  # Adjust based on font size
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame,
                 (text_x, text_y - text_height - 5),
                 (text_x + text_width, text_y + 5),
                 (0, 0, 0), -1)  # Black filled rectangle
    
    # Draw the text (white color)
    cv2.putText(frame, label, 
               (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (255, 255, 255), 1)  
    
def update_threshold(val):
    global conf_threshold
    conf_threshold = val / 100  # Convert trackbar value to range [0,1]

prev_positions = {}         #           currPos   boundary
def check_crossing_directionU(track_id, center_y, line_y):
    global prev_positions
    direction = None

    if track_id in prev_positions:
        prev_y = prev_positions[track_id]
     #prevPos is before line  &  currPos is after line          
        if (prev_y < line_y) and (line_y <= center_y):          # center_y >= line_y
            direction = 'down'
        elif (prev_y > line_y) and (center_y <= line_y):
            direction = 'up'

    prev_positions[track_id] = center_y
    return direction

prev_positions2 = {}
def check_crossing_directionD(track_id, center_y, line_y):
    global prev_positions2
    direction = None

    if track_id in prev_positions2:
        prev_y = prev_positions2[track_id]
     #prevPos is before line  &  currPos is after line          
        if (prev_y < line_y) and (center_y >= line_y):
            direction = 'down'
        elif prev_y > line_y and center_y <= line_y:
            direction = 'up'

    prev_positions2[track_id] = center_y
    return direction

def remove_from_cart(prod_id):
    # mqtt_client.publish(topic, "hi data", qos=1)
    product = products.get(prod_id)
    if product:
        data = f"{prod_id}&&&{product['name']}&&&-1&&&{product['price']}" 
        # data = f"{product['name']}&&&-1&&&{product['price']}"
        # with open("log.txt", "a") as f:
        #     f.write(f"{data}\n")
        try:
            mqtt_client.publish(topic, data, qos=1)
        except ssl.SSLError as e:
            print("SSL error occurred, reconnecting MQTT client...")
            mqtt_client.reconnect()
            mqtt_client.publish(topic, data, qos=1)
        time.sleep(0.1)

def add_to_cart(prod_id):
    # mqtt_client.publish(topic, "hi data", qos=1)
    product = products.get(prod_id)
    if product:
        data = f"{prod_id}&&&{product['name']}&&&1&&&{product['price']}" 
        # data = f"{product['name']}&&&1&&&{product['price']}"    #OR +1
        # with open("log.txt", "a") as f:
        #     f.write(f"{data}\n")
        try:
            mqtt_client.publish(topic, data, qos=1)
        except ssl.SSLError as e:
            print("SSL error occurred, reconnecting MQTT client...")
            mqtt_client.reconnect()
            mqtt_client.publish(topic, data, qos=1)
        time.sleep(0.1)



HOST = '0.0.0.0'
PORT = 8765


async def handler(websocket):
    print("Client connected.")
    try:
        async for frame in websocket:
            if isinstance(frame, bytes):
                await infere(frame)
            else:
                print(f"Received text message: {frame}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print("Error during handler:", e)


async def main():
    print(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    #**remove**
    cv2.namedWindow("YOLOv8 OBB Inference")
    cv2.createTrackbar("Confidence", "YOLOv8 OBB Inference", int(conf_threshold * 100), 100, update_threshold)
    
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")
        cv2.destroyAllWindows()