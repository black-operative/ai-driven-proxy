import socket
import struct
import pickle

PORT = 5000
HOST = "127.0.0.1"
MODEL = pickle.load(open("model.pkl", "rb"))
SOCKET_PATH = (HOST, PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(SOCKET_PATH)
sock.listen(1)

print("AI Model listening on port", PORT, "...")

conn, _ = sock.accept()

print("Proxy Server connection established")

while True:
    data = conn.recv(24)  # size of FeatureVector
    if not data:
        break

    features = struct.unpack("IIIQ", data)
    prediction = MODEL.predict([features])[0]
    confidence = max(MODEL.predict_proba([features])[0])

    response = struct.pack("Bf", prediction, confidence)
    conn.send(response)
