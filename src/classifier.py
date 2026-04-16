import tensorflow as tf
import numpy as np


class TFLiteClassifier:
    def __init__(self, model_path, num_threads=1):
        """
        A unified classifier for TFLite models.
        :param model_path: Path object from config.py
        """
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path), num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data: np.ndarray) -> int:
        """
        Runs inference on the provided normalized data.
        """
        # Ensure data is float32 and has the correct batch dimension
        input_tensor = np.array([input_data], dtype=np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        # Return the index of the highest probability
        return np.argmax(np.squeeze(output_data))
