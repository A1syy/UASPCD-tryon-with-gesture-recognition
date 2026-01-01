#udp_webcam_server.py
#!/usr/bin/env python3
"""
UDP Webcam Server - Optimized for Performance
Integrated with filter_ref.FilterEngine (headless) via command queue.

Features added:
- Integration point with FilterEngine (filter_ref.py). Server will call engine.process_frame(frame)
  and send filtered frames to clients.
- Command handling: clients can send text commands to server:
    - "REGISTER" / "UNREGISTER" (existing)
    - "SET_MASK <filename>"  -> set mask by filename relative to masks_folder (or path if no folder)
    - "SET_MASK_PATH <fullpath>" -> set mask by full filesystem path
  Commands are queued by the listener thread and executed in the broadcast thread to avoid
  Mediapipe multi-thread issues.
- Safe shutdown of engine on stop.
- Backwards compatible: if filter_ref not present, server still streams raw frames.

Notes:
- This version adds small robustness for 'masks' folder detection (tries "mask" and "masks")
  relative to the server script location, and normalizes to an absolute path.
"""

import os
import cv2
import socket
import struct
import threading
import time
import math
import queue
import traceback

# Try to import the FilterEngine (filter_ref.py). If missing, continue without filter.
try:
    from filter_ref import FilterEngine
except Exception as e:
    FilterEngine = None
    print("⚠️ filter_ref.FilterEngine not available:", e)


class UDPWebcamServer:
    def __init__(self, host='127.0.0.1', port=8888, masks_folder: str = None):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = set()
        self.camera = None
        self.running = False
        self.sequence_number = 0

        # Command queue: listener thread pushes commands here, broadcast thread executes them
        self.command_queue = queue.Queue()

        # Engine (created if FilterEngine available)
        self.engine = None

        # Normalize/resolve masks_folder: if provided and relative, try resolve relative to script dir.
        # If not provided, attempt to auto-detect "mask" or "masks" next to this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if masks_folder:
            # If given folder is relative, check relative to script dir
            if not os.path.isabs(masks_folder):
                candidate = os.path.join(script_dir, masks_folder)
                if os.path.isdir(candidate):
                    self.masks_folder = os.path.abspath(candidate)
                elif os.path.isdir(masks_folder):
                    # provided relative to cwd, still valid
                    self.masks_folder = os.path.abspath(masks_folder)
                else:
                    # keep as-is (maybe caller expects to use absolute path later)
                    self.masks_folder = masks_folder
            else:
                # absolute path given
                self.masks_folder = masks_folder if os.path.isdir(masks_folder) else masks_folder
        else:
            # auto-detect common folder names relative to script dir
            found = None
            for name in ("mask", "masks"):
                cand = os.path.join(script_dir, name)
                if os.path.isdir(cand):
                    found = cand
                    break
            if found:
                self.masks_folder = os.path.abspath(found)
                print(f"ℹ️ Auto-detected masks folder: {self.masks_folder}")
            else:
                self.masks_folder = None
                print("ℹ️ No masks folder auto-detected (mask/ masks). You can pass masks_folder to the server constructor.")

        # Optimized settings
        self.max_packet_size = 32768  # Reduced from 60KB to 32KB
        self.target_fps = 15  # Reduced from 30 to 15 FPS
        self.jpeg_quality = 40  # Reduced from 50 to 40
        self.frame_width = 480  # Reduced from 640
        self.frame_height = 360  # Reduced from 480

        # Performance monitoring
        self.frame_send_time = 1.0 / self.target_fps

    def initialize_camera(self):
        print("🎥 Initializing optimized camera...")
        # Use CAP_DSHOW on Windows for lower latency; if fails, fallback
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception:
            self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            # Set optimized resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            try:
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer (may not be supported everywhere)
            except Exception:
                pass

            ret, frame = self.camera.read()
            if ret and frame is not None:
                print(f"✅ Camera ready: {self.frame_width}x{self.frame_height} @ {self.target_fps}FPS")
                return True

        print("❌ Camera initialization failed")
        return False

    def start_server(self):
        if not self.initialize_camera():
            return

        # instantiate filter engine here if available
        if FilterEngine is not None:
            try:
                # pass masks_folder (can be None) into engine
                self.engine = FilterEngine(masks_folder=self.masks_folder, det_scale=0.75)
                print("🔧 FilterEngine initialized (filter_ref.py detected).")
            except Exception as e:
                self.engine = None
                print("⚠️ Failed to create FilterEngine:", e)
                traceback.print_exc()
        else:
            print("ℹ️ FilterEngine not available — streaming raw camera frames.")

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Increase send buffer for performance
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 655360)
        except Exception:
            pass
        self.server_socket.bind((self.host, self.port))

        print(f"🚀 Optimized UDP Server: {self.host}:{self.port}")
        print(f"📊 Settings: {self.frame_width}x{self.frame_height}, {self.target_fps}FPS, Q{self.jpeg_quality}")

        self.running = True

        # Start threads
        threading.Thread(target=self.listen_for_clients, daemon=True).start()
        threading.Thread(target=self._broadcast_frames, daemon=True).start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by KeyboardInterrupt")
        finally:
            self.stop_server()

    def listen_for_clients(self):
        """
        Listen for small UDP control packets from clients.
        Valid control messages (plain UTF-8 text):
          - "REGISTER"
          - "UNREGISTER"
          - "SET_MASK <filename>"
          - "SET_MASK_PATH <fullpath>"
        """
        self.server_socket.settimeout(1.0)

        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(2048)
                # try to decode as utf-8; if not text, ignore
                try:
                    message = data.decode('utf-8').strip()
                except Exception:
                    continue

                if not message:
                    continue

                # registration messages
                if message == "REGISTER":
                    if addr not in self.clients:
                        self.clients.add(addr)
                        print(f"✅ Client: {addr} (Total: {len(self.clients)})")
                    # ack
                    try:
                        self.server_socket.sendto("REGISTERED".encode('utf-8'), addr)
                    except Exception:
                        pass

                elif message == "UNREGISTER":
                    self.clients.discard(addr)
                    print(f"❌ Client left: {addr}")

                # SET_MASK commands: queue for broadcast thread to execute
                elif message.startswith("SET_MASK "):
                    arg = message[len("SET_MASK "):].strip()
                    if arg:
                        self.command_queue.put(("SET_MASK", arg, addr))
                        # immediate ack that command was received
                        try:
                            self.server_socket.sendto("SET_MASK_RECEIVED".encode('utf-8'), addr)
                        except Exception:
                            pass

                elif message.startswith("SET_MASK_PATH "):
                    arg = message[len("SET_MASK_PATH "):].strip()
                    if arg:
                        self.command_queue.put(("SET_MASK_PATH", arg, addr))
                        try:
                            self.server_socket.sendto("SET_MASK_PATH_RECEIVED".encode('utf-8'), addr)
                        except Exception:
                            pass

                elif message == "LIST_MASKS":
                    # respond with comma-separated list if masks_folder set
                    if self.masks_folder and os.path.isdir(self.masks_folder):
                        try:
                            files = [f for f in os.listdir(self.masks_folder) if os.path.isfile(os.path.join(self.masks_folder, f))]
                            payload = ",".join(files) if files else "NO_MASKS"
                        except Exception:
                            payload = "ERR_LIST"
                    else:
                        payload = "NO_FOLDER"
                    try:
                        self.server_socket.sendto(payload.encode('utf-8'), addr)
                    except Exception:
                        pass

                else:
                    # unknown text commands ignored (or extend here)
                    # print(f"🔍 Unknown message from {addr}: {message}")
                    pass

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Client listener error: {e}")
                    traceback.print_exc()

    def _handle_command_now(self, cmd: str, arg: str, addr):
        """
        Execute command IN BROADCAST THREAD. This avoids calling Mediapipe from the listener thread.
        """
        if not self.engine:
            # if no engine, reply error to requester
            try:
                self.server_socket.sendto("ERR_NO_ENGINE".encode('utf-8'), addr)
            except Exception:
                pass
            print("⚠️ Received command but FilterEngine not available.")
            return

        try:
            if cmd == "SET_MASK":
                # arg is filename, relative to masks_folder if set
                try:
                    ok = self.engine.set_mask(arg)
                    if ok:
                        self.server_socket.sendto(f"MASK_SET:{arg}".encode('utf-8'), addr)
                        print(f"🎭 Mask set to: {arg} (requested by {addr})")
                    else:
                        self.server_socket.sendto(f"ERR_SET_MASK:{arg}".encode('utf-8'), addr)
                except Exception as e:
                    try:
                        self.server_socket.sendto(f"ERR_SET_MASK:{arg}".encode('utf-8'), addr)
                    except Exception:
                        pass
                    print("❌ Error setting mask:", e)
                    traceback.print_exc()

            elif cmd == "SET_MASK_PATH":
                # arg is full path
                try:
                    ok = self.engine.set_mask_path(arg)
                    if ok:
                        self.server_socket.sendto(f"MASK_PATH_SET".encode('utf-8'), addr)
                        print(f"🎭 Mask set by path (requested by {addr}): {arg}")
                    else:
                        self.server_socket.sendto(f"ERR_SET_MASK_PATH".encode('utf-8'), addr)
                except Exception as e:
                    try:
                        self.server_socket.sendto(f"ERR_SET_MASK_PATH".encode('utf-8'), addr)
                    except Exception:
                        pass
                    print("❌ Error setting mask by path:", e)
                    traceback.print_exc()

            else:
                # unknown command
                try:
                    self.server_socket.sendto("ERR_UNKNOWN_CMD".encode('utf-8'), addr)
                except Exception:
                    pass

        except Exception:
            traceback.print_exc()

    def _broadcast_frames(self):
        last_frame_time = 0

        while self.running:
            # Execute queued commands first (so mask changes happen in this thread)
            try:
                while not self.command_queue.empty():
                    cmd, arg, addr = self.command_queue.get_nowait()
                    self._handle_command_now(cmd, arg, addr)
            except Exception:
                # ignore queue errors
                pass

            current_time = time.time()

            # If no clients, sleep but still handle queued commands above
            if len(self.clients) == 0:
                time.sleep(0.1)
                continue

            # Frame rate control
            if current_time - last_frame_time < self.frame_send_time:
                time.sleep(0.005)
                continue

            ret, frame = self.camera.read()
            if not ret:
                # try to continue; don't break to allow proper shutdown and commands
                time.sleep(0.05)
                continue

            # If filter engine is available, process frame through engine
            if self.engine:
                try:
                    # process_frame is called here in the broadcast thread (safe)
                    processed = self.engine.process_frame(frame)
                    if processed is not None:
                        out_frame = processed
                    else:
                        out_frame = frame
                except Exception as e:
                    # On error, fallback to raw frame and log
                    print("⚠️ FilterEngine processing error:", e)
                    traceback.print_exc()
                    out_frame = frame
            else:
                out_frame = frame

            # Encode with optimized settings
            try:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                result, encoded_img = cv2.imencode('.jpg', out_frame, encode_param)
            except Exception as e:
                print("⚠️ JPEG encode error:", e)
                traceback.print_exc()
                result = False
                encoded_img = None

            if result and encoded_img is not None:
                frame_data = encoded_img.tobytes()
                self.send_frame_to_clients(frame_data)
                last_frame_time = current_time

        # loop exit
        print("🛑 Broadcast thread exiting")

    def send_frame_to_clients(self, frame_data):
        if not frame_data or len(self.clients) == 0:
            return

        self.sequence_number = (self.sequence_number + 1) % 65536
        frame_size = len(frame_data)

        header_size = 12
        payload_size = self.max_packet_size - header_size
        total_packets = math.ceil(frame_size / payload_size)

        # Send to all clients efficiently
        for client_addr in self.clients.copy():
            try:
                for packet_index in range(total_packets):
                    start_pos = packet_index * payload_size
                    end_pos = min(start_pos + payload_size, frame_size)

                    header = struct.pack("!III", self.sequence_number, total_packets, packet_index)
                    udp_packet = header + frame_data[start_pos:end_pos]

                    self.server_socket.sendto(udp_packet, client_addr)

                # Less frequent logging
                if self.sequence_number % 60 == 1:  # Every 4 seconds at 15FPS
                    print(f"📤 Frame {self.sequence_number}: {frame_size//1024}KB → {len(self.clients)} clients")

            except Exception as e:
                print(f"❌ Send error {client_addr}: {e}")
                self.clients.discard(client_addr)

    def stop_server(self):
        print("⏹️ Stopping server...")
        self.running = False

        # Close socket and camera
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass

        # Close engine if present
        if self.engine:
            try:
                self.engine.close()
            except Exception:
                pass

        print("✅ Server stopped")


if __name__ == "__main__":
    print("=== Optimized UDP Webcam Server (with filter integration) ===")
    # No hardcoded folder here; UDPWebcamServer will try to autodetect "mask"/"masks" next to this script.
    server = UDPWebcamServer(masks_folder=None)
    server.start_server()
