from PyQt6 import QtWidgets
from data_collector import DataCollectionUI
from NeuralNetwork import DrivingNeuralNetwork
import numpy as np
import torch
import gc
import os

class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = DrivingNeuralNetwork()
        self.prediction_count = 0
        self.connection_lost = False
        
        dir = "C:\\Users\\kevin\\OneDrive\\Bureau\\RallyRobotPilot_2025\\scripts\\train_data"
        print(f"[INIT] Starting Neural Network Autopilot")
        files = [
            os.path.join(dir,f)
            for f in os.listdir(dir)
            if f.endswith(".npz")
        ]
        self.model.train_model(
            files,
            learning_rate=0.0005,
            epochs=30,
            batch_size=64
        )

        
        # Set eval mode once
        self.model.eval()
        torch.set_grad_enabled(False)
        
        print("[+] Model ready!\n")

    def nn_infer(self, message):
        try:
            self.prediction_count += 1
            
            # Extract features
            features = np.array([
                float(message.car_speed),
                float(message.car_angle),
                *message.raycast_distances
            ], dtype=np.float32).copy()
            
            # Predict (returns list of tuples: [('forward', True), ('back', False), ...])
            predictions = self.model.predict(features, threshold=0.3)
            
            # GC every 5 predictions
            if self.prediction_count % 5 == 0:
                gc.collect()
            
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    import sys
    
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    print("\n" + "="*60)
    print("  RALLY ROBOT PILOT - NEURAL NETWORK AUTOPILOT")
    print("="*60 + "\n")

    app = QtWidgets.QApplication(sys.argv)

    try:
        nn_brain = ExampleNNMsgProcessor()
        data_window = DataCollectionUI(nn_brain.process_message)
        data_window.show()
        
        print("[MAIN] Waiting for game connection...\n")
        
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(app.exec())