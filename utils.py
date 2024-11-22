import xml.etree.ElementTree as ET
import numpy as np
import os
import imageio
from typing import List, Tuple, Callable, Optional


def save_tif(sequence: np.ndarray, output_path: str) -> None:
    try:
        imageio.volwrite(output_path, sequence, format="TIFF")
        print(f"Saved sequence as {output_path}")
    except Exception as e:
        print(f"Error saving TIFF: {e}")


class ImageProcessor:
    """
    A class for processing and visualizing sequences of medical images.
    """

    # Class attributes (g(x) coefficients)
    K1 = 8000
    K2 = 0.025
    K3 = -40
    K4 = 0

    def __init__(
        self,
        sequence_path: str,
        window: int,
        level: int,
        mask_idx: int = 0,
        filter_type: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the ImageProcessor.

        Parameters:
        - sequence_path: path to the sequence file
        - window: window width for image adjustment
        - level: level for image adjustment
        - mask_idx: index of the mask in the sequence
        - filter_type:  denoising filter function (optional)
        """
        self.sequence_path: str = sequence_path
        self.mask_idx: int = mask_idx
        self.filter: Optional[Callable] = filter_type

        self.num_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.sequence: Optional[np.ndarray] = None
        self.filtered_normal_sequence: Optional[np.ndarray] = None
        self.filtered_dsa_sequence: Optional[np.ndarray] = None
        self.dsa_sequence: Optional[np.ndarray] = None
        self.window: int = window
        self.level: int = level

    def _get_sequence_dimensions(self) -> Tuple[int, int, int]:
        xml_path = os.path.splitext(self.sequence_path)[0] + ".xml"

        if not os.path.exists(xml_path):
            print(f"XML file '{xml_path}' doesn't exist.")
            self.width = int(input("Please enter the image width: "))
            self.height = int(input("Please enter the image height: "))
            self.num_frames = int(input("Please enter the number of frames: "))
        else:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                frames = root.findall("./frame")
                self.num_frames = len(frames)

                first_frame = root.find("./frame")
                self.width = int(first_frame.find("imgWidth").text)
                self.height = int(first_frame.find("imgHeight").text)
            except Exception as e:
                print(f"Error reading XML file: {e}")
                raise
        return self.num_frames, self.width, self.height

    def load_sequence(self) -> np.ndarray:
        self.num_frames, self.width, self.height = self._get_sequence_dimensions()
        try:
            with open(self.sequence_path, "rb") as file:
                raw_sequence = np.fromfile(file, dtype=np.uint16)
                self.sequence = raw_sequence.reshape(
                    (self.num_frames, self.height, self.width)
                )
        except Exception as e:
            print(f"Error loading sequence: {e}")
            raise
        return self.sequence

    def apply_filter(self, sequence: np.ndarray, filter_args: list = None) -> np.ndarray:
        if self.filter is None:
            raise ValueError("No filter function provided.")

        filtered_sequence = np.empty_like(sequence, dtype=np.uint16)
        for i, frame in enumerate(sequence):
            filtered_sequence[i] = self.filter(frame, *filter_args).astype(np.uint16)

        return filtered_sequence


    def perform_dsa(self, filter_args: list = None) -> np.ndarray:
        def g(x, k1=self.K1, k2=self.K2, k3=self.K3, k4=self.K4):
            x = x.astype(np.float64)
            gx = k1 * np.log10(k2 * (x - k3)) - k4
            return gx  # float64

        gx1 = g(self.sequence[self.mask_idx])
        self.dsa_sequence = np.empty_like(self.sequence, dtype=np.uint16)

        for i, frame in enumerate(self.sequence):
            gxn = g(frame)

            fy = gxn - gx1 + 32768  # Normalize to uint16 range
            if self.filter is not None:
                fy = self.filter(fy, *filter_args)

            self.dsa_sequence[i] = np.clip(fy, 0, 65535).astype(np.uint16)

        return self.dsa_sequence


    def adjust_wl(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust window and level for an image.

        Parameters:
        - image: input image

        Returns:
        adjusted image
        """
        min_value = self.level
        max_value = self.level + self.window

        # Clip pixel values within the specified window
        adjusted_image = np.clip(image, min_value, max_value)

        # Normalize the pixel values within the window to the range [0, 1]
        adjusted_image = (adjusted_image - min_value) / (max_value - min_value)

        # Scale the normalized values to the 8-bit range [0, 255]
        adjusted_image_8bit = (adjusted_image * 255).astype(np.uint8)

        return adjusted_image_8bit

    def collect_frames(self, sequence: np.ndarray, rescale: bool = False):
        frames_list = [
            self.adjust_wl(frame) if rescale else frame for frame in sequence
        ]

        return frames_list

    def estimate_noise(self) -> float:
        std_devs = [np.std(frame) for frame in self.sequence]
        average_noise = np.mean(std_devs)
        return average_noise
