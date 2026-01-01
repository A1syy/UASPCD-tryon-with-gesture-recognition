# #!/usr/bin/env python3
# """
# UDP Webcam Server - Optimized for Performance
# """

# import cv2
# import socket
# import struct
# import threading
# import time
# import math

# class UDPWebcamServer:
#     def __init__(self, host='127.0.0.1', port=8888):
#         self.host = host
#         self.port = port
#         self.server_socket = None
#         self.clients = set()
#         self.camera = None
#         self.running = False
#         self.sequence_number = 0
        
#         # Optimized settings
#         self.max_packet_size = 32768  # Reduced from 60KB to 32KB
#         self.target_fps = 15  # Reduced from 30 to 15 FPS
#         self.jpeg_quality = 40  # Reduced from 50 to 40
#         self.frame_width = 480  # Reduced from 640
#         self.frame_height = 360  # Reduced from 480
        
#         # Performance monitoring
#         self.frame_send_time = 1.0 / self.target_fps
        
#     def initialize_camera(self):
#         print("🎥 Initializing optimized camera...")
#         self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if self.camera.isOpened():
#             # Set optimized resolution
#             self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
#             self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
#             self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
#             self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
#             ret, frame = self.camera.read()
#             if ret and frame is not None:
#                 print(f"✅ Camera ready: {self.frame_width}x{self.frame_height} @ {self.target_fps}FPS")
#                 return True
        
#         print("❌ Camera initialization failed")
#         return False
    
#     def start_server(self):
#         if not self.initialize_camera():
#             return
        
#         self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 655360)  # 640KB send buffer
#         self.server_socket.bind((self.host, self.port))
        
#         print(f"🚀 Optimized UDP Server: {self.host}:{self.port}")
#         print(f"📊 Settings: {self.frame_width}x{self.frame_height}, {self.target_fps}FPS, Q{self.jpeg_quality}")
        
#         self.running = True
        
#         # Start threads
#         threading.Thread(target=self.listen_for_clients, daemon=True).start()
#         threading.Thread(target=self._broadcast_frames, daemon=True).start()
        
#         try:
#             while self.running:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             print("\n🛑 Server stopped")
#         finally:
#             self.stop_server()
    
#     def listen_for_clients(self):
#         self.server_socket.settimeout(1.0)
        
#         while self.running:
#             try:
#                 data, addr = self.server_socket.recvfrom(1024)
#                 message = data.decode('utf-8')
                
#                 if message == "REGISTER":
#                     if addr not in self.clients:
#                         self.clients.add(addr)
#                         print(f"✅ Client: {addr} (Total: {len(self.clients)})")
                    
#                     self.server_socket.sendto("REGISTERED".encode('utf-8'), addr)
                
#                 elif message == "UNREGISTER":
#                     self.clients.discard(addr)
#                     print(f"❌ Client left: {addr}")
                    
#             except socket.timeout:
#                 continue
#             except Exception as e:
#                 if self.running:
#                     print(f"⚠️ Client error: {e}")
    
#     def _broadcast_frames(self):
#         last_frame_time = 0
        
#         while self.running:
#             current_time = time.time()
            
#             # Skip if no clients
#             if len(self.clients) == 0:
#                 time.sleep(0.1)
#                 continue
            
#             # Frame rate control
#             if current_time - last_frame_time < self.frame_send_time:
#                 time.sleep(0.01)
#                 continue
            
#             ret, frame = self.camera.read()
#             if not ret:
#                 break
            
#             # Encode with optimized settings
#             encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
#             result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
#             if result:
#                 frame_data = encoded_img.tobytes()
#                 self.send_frame_to_clients(frame_data)
#                 last_frame_time = current_time
    
#     def send_frame_to_clients(self, frame_data):
#         if not frame_data or len(self.clients) == 0:
#             return
        
#         self.sequence_number = (self.sequence_number + 1) % 65536
#         frame_size = len(frame_data)
        
#         header_size = 12
#         payload_size = self.max_packet_size - header_size
#         total_packets = math.ceil(frame_size / payload_size)
        
#         # Send to all clients efficiently
#         for client_addr in self.clients.copy():
#             try:
#                 for packet_index in range(total_packets):
#                     start_pos = packet_index * payload_size
#                     end_pos = min(start_pos + payload_size, frame_size)
                    
#                     header = struct.pack("!III", self.sequence_number, total_packets, packet_index)
#                     udp_packet = header + frame_data[start_pos:end_pos]
                    
#                     self.server_socket.sendto(udp_packet, client_addr)
                
#                 # Less frequent logging
#                 if self.sequence_number % 60 == 1:  # Every 4 seconds at 15FPS
#                     print(f"📤 Frame {self.sequence_number}: {frame_size//1024}KB → {len(self.clients)} clients")
                    
#             except Exception as e:
#                 print(f"❌ Send error {client_addr}: {e}")
#                 self.clients.discard(client_addr)
    
#     def stop_server(self):
#         print("⏹️ Stopping server...")
#         self.running = False
        
#         if self.server_socket:
#             self.server_socket.close()
#         if self.camera:
#             self.camera.release()
        
#         print("✅ Server stopped")

# if __name__ == "__main__":
#     print("=== Optimized UDP Webcam Server ===")
#     server = UDPWebcamServer()
#     server.start_server()