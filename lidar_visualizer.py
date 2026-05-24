#!/usr/bin/env python3
"""
Fixed LiDAR visualizer for STYTJ02YM
Based on actual data format: AA 00 43 01 61 AD ...
"""

import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import sys
import logging
from collections import deque
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class LidarDecoder:
    """Fixed decoder for STYTJ02YM LiDAR"""
    
    def __init__(self):
        self.buffer = bytearray()
        self.current_scan = {}
        self.full_scan_complete = False
        self.last_scan_time = time.time()
        self.frame_count = 0
        self.scan_count = 0
        
    def find_frames(self, data):
        """Find and extract complete frames from data stream"""
        self.buffer.extend(data)
        frames = []
        
        i = 0
        while i < len(self.buffer):
            # Look for header (0xAA)
            if self.buffer[i] == 0xAA:
                # Need at least 3 bytes to get length
                if i + 2 >= len(self.buffer):
                    break
                
                # Get length (byte at offset 2)
                frame_len = self.buffer[i + 2]
                
                # Sanity check on length
                if frame_len < 4 or frame_len > 200:
                    # Invalid length, skip this byte
                    i += 1
                    continue
                
                # Check if we have the complete frame
                if i + frame_len <= len(self.buffer):
                    frame = self.buffer[i:i+frame_len]
                    frames.append(frame)
                    i += frame_len
                else:
                    # Wait for more data
                    break
            else:
                i += 1
        
        # Keep any remaining incomplete data in buffer
        if i < len(self.buffer):
            self.buffer = self.buffer[i:]
        else:
            self.buffer.clear()
            
        return frames
    
    def decode_measurement_frame(self, frame):
        """Decode measurement frame (type 0xAD)"""
        if len(frame) < 10:
            return None
        
        # Verify it's a measurement frame
        if frame[4] != 0x61 or frame[5] != 0xAD:
            return None
        
        # Payload starts at byte 8, ends before last 2 bytes (CRC)
        payload = frame[8:-2]
        
        if len(payload) < 5:
            return None
        
        # Parse payload header
        rpm_scaled = payload[0]
        rpm = rpm_scaled * 3
        
        # Reserved/offset bytes (skip payload[1], payload[2])
        offset_angle = ((payload[3] << 8) | payload[4]) * 0.01
        
        # Remaining bytes are samples (each sample = 3 bytes: quality, distance_high, distance_low)
        sample_data = payload[5:]
        num_samples = len(sample_data) // 3
        
        samples = []
        
        # Each message covers 24 degrees (360/15 = 24)
        # The start angle is in the payload, and samples are evenly spaced
        start_angle = offset_angle  # Use offset angle as start angle
        
        for i in range(num_samples):
            idx = i * 3
            if idx + 2 >= len(sample_data):
                break
                
            quality = sample_data[idx]
            distance_raw = (sample_data[idx + 1] << 8) | sample_data[idx + 2]
            distance_mm = distance_raw * 0.25
            
            # Calculate angle for this sample
            # Each message covers 24 degrees, so increment per sample
            angle_increment = 24.0 / num_samples
            angle = start_angle + (i * angle_increment)
            angle = angle % 360.0
            
            # Filter valid readings (typical range: 0.1m to 12m)
            if 100 < distance_mm < 12000 and quality > 0:
                samples.append({
                    'angle': angle,
                    'distance': distance_mm,
                    'quality': quality
                })
        
        return {
            'type': 'measurement',
            'rpm': rpm,
            'start_angle': start_angle,
            'num_samples': num_samples,
            'samples': samples
        }
    
    def decode_health_frame(self, frame):
        """Decode health frame (type 0xAE)"""
        if len(frame) < 9:
            return None
        
        # Verify it's a health frame
        if frame[4] != 0x61 or frame[5] != 0xAE:
            return None
        
        # Payload starts at byte 8
        if len(frame) > 8:
            rpm_scaled = frame[8]
            rpm = rpm_scaled * 3
            return {
                'type': 'health',
                'rpm': rpm
            }
        return None
    
    def process_data(self, data):
        """Process incoming raw data"""
        frames = self.find_frames(data)
        messages = []
        
        for frame in frames:
            self.frame_count += 1
            
            # Try to decode as measurement frame first
            msg = self.decode_measurement_frame(frame)
            if msg:
                messages.append(msg)
                continue
            
            # Try health frame
            msg = self.decode_health_frame(frame)
            if msg:
                messages.append(msg)
                
        return messages
    
    def update_scan(self, measurement_msg):
        """Update current scan with new measurement data"""
        if not measurement_msg['samples']:
            return False
        
        # Add all samples to current scan
        for sample in measurement_msg['samples']:
            angle = sample['angle']
            distance = sample['distance']
            # Keep the best quality reading for each angle
            if angle not in self.current_scan or sample['quality'] > self.current_scan[angle]['quality']:
                self.current_scan[angle] = {
                    'distance': distance,
                    'quality': sample['quality']
                }
        
        # Check if we have a complete scan (points covering full circle)
        if len(self.current_scan) > 100:  # Minimum points for a scan
            angles = list(self.current_scan.keys())
            angle_range = max(angles) - min(angles)
            
            # If we have points covering most of the circle
            if angle_range > 350 or (self.current_scan and (time.time() - self.last_scan_time) > 0.5):
                # Complete scan detected
                self.scan_count += 1
                logger.info(f"Scan #{self.scan_count}: {len(self.current_scan)} points, RPM: {measurement_msg['rpm']}")
                
                # Create a copy of the scan for plotting
                scan_copy = {angle: self.current_scan[angle]['distance'] 
                           for angle in self.current_scan}
                
                # Reset for next scan
                self.current_scan = {}
                self.last_scan_time = time.time()
                
                return scan_copy
        
        return None

class LidarVisualizer:
    """Real-time polar plot visualization"""
    
    def __init__(self, max_range_mm=8000):
        self.max_range = max_range_mm / 1000.0
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.scan_data = None
        self.new_data_available = False
        
        # Setup plot
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rmax(self.max_range)
        self.ax.set_rticks([1, 2, 3, 4, 5, 6, 7, 8])
        self.ax.set_rlabel_position(22.5)
        self.ax.grid(True)
        self.ax.set_title("LiDAR Scan", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Distance (meters)", fontsize=12)
        
        # Info text
        self.info_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Scatter plot
        self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=8, alpha=0.7)
        self.cbar = plt.colorbar(self.scatter, ax=self.ax, label='Distance (m)')
        
    def update_plot(self, scan_data, scan_count, frame_count):
        """Update plot with new scan data"""
        if not scan_data or len(scan_data) == 0:
            return
        
        # Convert to numpy arrays
        angles_rad = np.radians(list(scan_data.keys()))
        distances_m = np.array(list(scan_data.values())) / 1000.0
        
        # Filter points
        mask = distances_m <= self.max_range
        angles_rad = angles_rad[mask]
        distances_m = distances_m[mask]
        
        if len(angles_rad) == 0:
            return
        
        # Update scatter plot
        self.scatter.set_offsets(np.column_stack([angles_rad, distances_m]))
        self.scatter.set_array(distances_m)
        
        # Update colorbar
        self.scatter.set_clim(vmin=0, vmax=self.max_range)
        self.cbar.update_normal(self.scatter)
        
        # Update info text
        info_str = f"Points: {len(scan_data)}\nScans: {scan_count}\nFrames: {frame_count}"
        self.info_text.set_text(info_str)
        
        # Redraw
        self.fig.canvas.draw_idle()
        self.new_data_available = False

def main():
    """Main function"""
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUD_RATE = 115200
    
    print("\n" + "="*60)
    print("STYTJ02YM LiDAR Visualizer")
    print("="*60)
    print(f"Port: {SERIAL_PORT}")
    print(f"Baud rate: {BAUD_RATE}")
    print("Press Ctrl+C to stop\n")
    
    # Initialize
    decoder = LidarDecoder()
    visualizer = LidarVisualizer(max_range_mm=8000)
    
    # Shared variables for thread communication
    scan_count = 0
    frame_count = 0
    
    def data_thread():
        nonlocal scan_count, frame_count
        
        # Initialize stats timer inside the thread
        last_stats_time = time.time()
        
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) as ser:
                logger.info(f"Connected to {SERIAL_PORT}")
                
                while True:
                    # Read data
                    data = ser.read(4096)
                    
                    if data:
                        # Process frames
                        messages = decoder.process_data(data)
                        
                        for msg in messages:
                            frame_count += 1
                            
                            if msg['type'] == 'measurement':
                                # Update scan with new data
                                complete_scan = decoder.update_scan(msg)
                                
                                if complete_scan:
                                    scan_count += 1
                                    visualizer.new_data_available = True
                                    visualizer.scan_data = complete_scan
                                    
                                    # Log progress
                                    if scan_count % 10 == 0:
                                        logger.info(f"Scan {scan_count}: {len(complete_scan)} points")
                            
                            elif msg['type'] == 'health':
                                logger.debug(f"RPM: {msg['rpm']}")
                    
                    # Periodic stats
                    if time.time() - last_stats_time > 30:
                        logger.info(f"Stats - Scans: {scan_count}, Frames: {frame_count}")
                        last_stats_time = time.time()
                    
                    time.sleep(0.001)
                    
        except serial.SerialException as e:
            logger.error(f"Serial error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Start data thread
    thread = threading.Thread(target=data_thread, daemon=True)
    thread.start()
    
    # Animation update
    def animate(frame):
        if visualizer.new_data_available and visualizer.scan_data:
            visualizer.update_plot(visualizer.scan_data, scan_count, frame_count)
        return visualizer.scatter,
    
    # Run visualization
    ani = FuncAnimation(visualizer.fig, animate, interval=50, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()