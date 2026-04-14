import torch


class CircularTensorNMS:
    def __init__(
        self,
        threshold: float = 0.95,
    ):
        """
        Initialize the CircularTensorNMS class.

        Args:
            threshold (float): The minimum value to consider a valid peak.
            min_peak_distance (int): The minimum distance between peaks (in terms of rays).
        """
        self.threshold = threshold
        self.peaks_pixels_at_distance = {
            "min": {"distance": 0.6, "pixel_width": 33},
            "max": {"distance": 6.0, "pixel_width": 3},
        }  # peaks are modelled as waist of people at different distances

    def find_connected_components_circular(self, boolean_array: torch.Tensor) -> list[tuple[int, list[int]]]:
        """
        Finds connected components in a boolean circular array and returns the indices
        of each connected component as a list of lists.

        Args:
            boolean_array (torch.Tensor): 1D tensor of boolean values (dtype=torch.bool).

        Returns:
            list of lists: Each list contains the indices of a connected component.
        """
        n = boolean_array.size(0)  # Get the length of the boolean array
        if n == 0:
            return []  # No components in an empty array

        components = []  # List to store the connected components

        # Find transitions (True if there is a transition)
        transitions = torch.logical_xor(boolean_array, torch.roll(boolean_array, 1, dims=-1))
        transitions_starts = torch.logical_and(transitions, boolean_array)
        transition_starts_indeces = torch.nonzero(transitions_starts, as_tuple=True)[0].tolist()
        transitions_ends = torch.roll(torch.logical_and(transitions, ~boolean_array), -1)
        transition_ends_indeces = torch.nonzero(transitions_ends, as_tuple=True)[0].tolist()

        while len(transition_starts_indeces) > 0:
            if transition_starts_indeces[0] > transition_ends_indeces[0]:
                start = transition_starts_indeces.pop(-1)
                end = transition_ends_indeces.pop(0)
                # This means there is a connected component that wraps around the array
                indeces = list(range(start, n)) + list(range(0, end + 1))
                center = int((start + end + n) // 2 % n)
            else:
                start = transition_starts_indeces.pop(0)
                end = transition_ends_indeces.pop(0)
                center = int((start + end) // 2)
                indeces = list(range(start, end + 1))

            components.append((center, indeces))

        return components

    def connected_components_nms(self, tensor_dict: dict):
        """
        Apply thresholding and return a dictionary with peaks only.

        Args:
            tensor_dict (dict): A dictionary with keys 'presence', 'distance', 'cosine', 'sine',
                                each containing a tensor of shape (N_timestamps, N_rays).

        Returns:
            dict: A dictionary with the same structure as the input, but with zero arrays except at peaks.
        """
        presence_probs = tensor_dict["presence"].clone()  # Extract predicted presence probabilities
        presence_masks = presence_probs >= self.threshold

        output_dict = {key: torch.zeros_like(value) for key, value in tensor_dict.items()}

        for t in range(presence_masks.shape[0]):
            connected_components = self.find_connected_components_circular(presence_masks[t])
            for center, indeces in connected_components:
                for key in tensor_dict.keys():
                    mean_value = tensor_dict[key][t, indeces].mean()
                    output_dict[key][t, center] = mean_value
        return output_dict

    def iterative_peak_nms(self, tensor_dict: dict):
        """
        Apply iterative peak selection with suppression and return a dictionary with peaks only.

        Args:
            tensor_dict (dict): A dictionary with keys 'presence', 'distance', 'cosine', 'sine',
                                each containing a tensor of shape (N_timestamps, N_rays).

        Returns:
            dict: A dictionary with the same structure as the input, but with zero arrays except at peaks.
        """
        presence_probs_masked = tensor_dict["presence"].clone()  # Extract predicted presence probabilities

        presence_probs_masked[presence_probs_masked < self.threshold] = 0

        output_dict = {key: torch.zeros_like(value) for key, value in tensor_dict.items()}

        for t in range(presence_probs_masked.shape[0]):
            probs = presence_probs_masked[t].clone()

            while True:
                max_value, max_idx = probs.max(dim=0)

                if max_value < self.threshold:
                    break

                # Create a mean mask around the selected peak
                mean_mask = torch.arange(-1, 2, 1, device=probs.device)
                mean_mask = (max_idx + mean_mask) % probs.size(0)

                peak_distance = tensor_dict["distance"][t][mean_mask].mean()

                suppress_mask_dimension = self.peaks_pixels_at_distance["min"]["pixel_width"] + int(
                    (
                        self.peaks_pixels_at_distance["max"]["pixel_width"]
                        - self.peaks_pixels_at_distance["min"]["pixel_width"]
                    )
                    * (peak_distance - self.peaks_pixels_at_distance["min"]["distance"])
                    / (
                        self.peaks_pixels_at_distance["max"]["distance"]
                        - self.peaks_pixels_at_distance["min"]["distance"]
                    )
                )
                suppress_mask_dimension = max(
                    suppress_mask_dimension, self.peaks_pixels_at_distance["max"]["pixel_width"]
                )

                suppress_mask = torch.arange(
                    -int(suppress_mask_dimension / 2.0), int(suppress_mask_dimension / 2.0) + 1, device=probs.device
                )
                suppress_mask = (max_idx + suppress_mask) % probs.size(0)

                probs[suppress_mask] = 0

                for key in tensor_dict.keys():
                    # 3 rays around the peak are set to the mean value
                    mean_value = tensor_dict[key][t, mean_mask].mean()
                    output_dict[key][t, max_idx] = mean_value

        return output_dict


if __name__ == "__main__":
    # Initialize the CircularTensorNMS instance
    nms = CircularTensorNMS(threshold=0.9)

    # Mock input tensor dictionary with longer tensors and peaks at the intersection of start and end
    tensor_dict = {
        "presence": torch.tensor(
            [
                [0.98, 0.92, 0.5, 0.1, 0.2, 0.95, 0.99, 0.98, 0.4, 0.0, 0.1, 0.97],
                [0.97, 0.98, 0.92, 0.5, 0.1, 0.2, 0.95, 0.99, 0.98, 0.4, 0.0, 0.1],
            ]
        ),
        "distance": torch.tensor(
            [
                [1.5, 1.51, 0.0, 0.0, 0.0, 3.02, 3.0, 3.14, 0.10, 0.0, 0.0, 1.48],
                [1.48, 1.5, 1.51, 0.0, 0.0, 0.0, 3.02, 3.0, 3.14, 0.10, 0.0, 0.0],
            ]
        ),
        "cosine": torch.tensor(
            [
                [1.0, 0.99, 0.0, 0.0, 0.0, -1, -0.99, -0.98, 0.0, 0.0, 0.0, 0.99],
                [0.99, 1.0, 0.99, 0.0, 0.0, 0.0, -1, -0.99, -0.98, 0.0, 0.0, 0.0],
            ]
        ),
        "sine": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    }
    print("Input Tensor:")
    for key, value in tensor_dict.items():
        print(f"{key}:\n{value}")

    # Perform connected components NMS
    output_connected = nms.connected_components_nms(tensor_dict)

    print("Connected Components NMS Output:")
    for key, value in output_connected.items():
        print(f"{key}:\n{value}")

    # Perform iterative peak NMS
    output_iterative = nms.iterative_peak_nms(tensor_dict)

    print("\nIterative Peak NMS Output:")
    for key, value in output_iterative.items():
        print(f"{key}:\n{value}")
