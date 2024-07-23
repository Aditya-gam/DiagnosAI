import unittest
import torch
from src.model import ResNet50Model
from src.dataset import prepare_test_loader, load_data
from src.utils import initialize_model
import src.config as config


class TestModelPerformance(unittest.TestCase):
    """
    This class is a test suite for verifying the performance of the ResNet50 model on test data.
    It uses Python's unittest framework for setting up the test environment, running the tests,
    and providing assertions.
    """

    @classmethod
    def setUpClass(cls):
        """
        Class method to prepare the environment before executing tests. It loads the test dataset,
        initializes the model, and loads the trained model weights. This method runs once for the 
        entire class, setting up shared resources.
        """
        # Load the dataset and retrieve only the test data and labels.
        _, test_data, all_labels = load_data()

        # Prepare the DataLoader for the test set.
        cls.test_loader = prepare_test_loader(test_data, all_labels)

        # Initialize the ResNet50 model with the correct number of classes and move it to the appropriate device.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.model, _, _, _ = initialize_model(
            device, num_classes=len(all_labels))

        # Load pre-trained weights from a saved path and move the model to the assigned device.
        cls.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        cls.model.to(device)
        cls.device = device

    def test_model_output(self):
        """
        Tests whether the model can produce outputs matching the batch size of the inputs,
        ensuring that it can process data without shape mismatches or runtime errors.
        """
        with torch.no_grad():  # Disables gradient calculation to save memory and computations
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # Check if the number of outputs matches the number of input samples
                self.assertEqual(outputs.shape[0], labels.shape[0])
                break  # Limit testing to the first batch for efficiency

    def test_model_accuracy(self):
        """
        Test the model's accuracy to ensure it meets a specified performance threshold, such as 70% accuracy.
        This is an integration test to evaluate the end-to-end performance of the model.
        """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Again, disable gradients for testing
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy: {accuracy}")
        # Assert that the accuracy is greater than 70%
        self.assertGreater(accuracy, 0.7)


if __name__ == '__main__':
    unittest.main()
