import unittest
import torch
from src.model import ResNet50Model
from src.dataset import prepare_test_loader, load_data
from src.utils import initialize_model
import src.config as config


class TestModelPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data
        _, test_data, all_labels = load_data()
        cls.test_loader = prepare_test_loader(test_data, all_labels)

        # Initialize the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.model, _, _, _ = initialize_model(
            device, num_classes=len(all_labels))

        # Load trained model weights
        cls.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        cls.model.to(device)
        cls.device = device

    def test_model_output(self):
        """ Test that the model is able to produce outputs. """
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.assertEqual(outputs.shape[0], labels.shape[0])
                break  # Test with the first batch only

    def test_model_accuracy(self):
        """ Optionally, test for model's performance accuracy, assuming a threshold. """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy: {accuracy}")
        # Assume expected accuracy is at least 70%
        self.assertGreater(accuracy > 0.7)


if __name__ == '__main__':
    unittest.main()
