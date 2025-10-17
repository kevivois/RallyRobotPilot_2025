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
        
        dir = "./scripts/train_data_ana"
        print(f"[INIT] Starting Neural Network Autopilot")
        files = [
            os.path.join(dir,f)
            for f in os.listdir(dir)
            if f.endswith(".npz")
        ]
        dir_me = "./scripts/train_data"
        files_me = [
            os.path.join(dir_me,f)
            for f in os.listdir(dir_me)
            if f.endswith(".npz")
        ]
        
        dir_louis = "./scripts/train_data_louis"
        files_louis = [
            os.path.join(dir_louis,f)
            for f in os.listdir(dir_louis)
            if f.endswith(".zip")
        ]
        
    
        self.model.train_model(
            files_louis,
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
                *message.raycast_distances
            ], dtype=np.float32).copy()
            
            # Predict (returns list of tuples: [('forward', True), ('back', False), ...])
            predictions = self.model.predict(features)
            
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