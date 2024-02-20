from sys import path

src_path = 'src'
path.insert(0, src_path)

from load_data import HandwrittenDigitMNIST

class TestHandwrittenDigitMNIST:
    # Data files' paths
    test_images_filepath = "data/t10k-images-idx3-ubyte.gz"
    test_labels_filepath = "data/t10k-labels-idx1-ubyte.gz"

    def test_dataset_size(self):
        # Arrangement
        expected_size = 10000

        # Action
        test_data = HandwrittenDigitMNIST(
            self.test_images_filepath,
            self.test_labels_filepath)

        # Assertion
        assert len(test_data) == expected_size

    def test_limit_dataset_size(self):
        # Arrangement
        items_count = 10

        # Action
        test_data = HandwrittenDigitMNIST(
            self.test_images_filepath,
            self.test_labels_filepath,
            items_count=items_count)

        # Assertion
        assert len(test_data) == items_count
