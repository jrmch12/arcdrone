"""
PlotJuggler logging utility for real-time data visualization.
"""
import zmq
import json
import time
import subprocess
import threading


class PlotJugglerLogger:
    """
    Stream data to PlotJuggler via ZMQ for real-time visualization.
    Optionally auto-launches PlotJuggler with a saved layout.
    """
    def __init__(self, port=9873, layout_file=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.layout_file = layout_file
        print(f"PlotJuggler logger streaming on port {port}")
        
        # Auto-launch PlotJuggler with layout if provided
        if self.layout_file:
            self._launch_plotjuggler()
        
        # Give ZMQ and PlotJuggler time to establish connection
        time.sleep(0.5)
    
    def _launch_plotjuggler(self):
        """Start PlotJuggler automatically with saved layout in background thread."""
        def start_pj():
            try:
                cmd = ['plotjuggler', '--layout', self.layout_file]
                print(f"Launching PlotJuggler with layout: {self.layout_file}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"PlotJuggler start error: {e}")
            except FileNotFoundError:
                print("PlotJuggler not found in PATH. Install with: sudo apt install plotjuggler")
        
        # Start in separate daemon thread (won't block main program)
        thread = threading.Thread(target=start_pj, daemon=True)
        thread.start()
    
    def log(self, data_dict):
        """
        Send data to PlotJuggler.
        
        Args:
            data_dict: Dictionary with string keys and numeric values
        """
        try:
            # Add timestamp
            data_dict["timestamp"] = time.time()
            
            # Convert to JSON and send
            message = json.dumps(data_dict)
            self.socket.send_string(message)
        except Exception as e:
            # Don't crash the main script if logging fails
            print(f"PlotJuggler logging error: {e}")
    
    def close(self):
        """Clean up ZMQ resources."""
        try:
            self.socket.close()
            self.context.term()
        except Exception as e:
            print(f"Error closing PlotJuggler logger: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



# ======== HOW TO USE ========

# ==== Script Pattern

# # Initialize once in your script
# logger = PlotJugglerLogger(port=9872)  # Change port if needed

# # In your loop, log data as nested dictionaries
# data_packet = {
#     'timestamp': current_time,
#     'sensors': {
#         'motor_pos': [1.2, 3.4, 5.6],
#         'motor_current': [0.1, 0.2, 0.3]
#     },
#     'targets': {
#         'desired_pos': [1.0, 3.0, 5.0]
#     },
#     'inference': {
#         'action': action.tolist(),  # Convert JAX arrays
#         'inference_time': 0.003
#     }
# }
# logger.log(data_packet)


# ==== To Set up in PlotJuggler

# Port: 9872 (or your chosen port)
# Message Protocol: JSON