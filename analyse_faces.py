import os
from dotenv import load_dotenv
from typing import List
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Azure SDK imports
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection03,
    FaceDetectionResult,
)
from azure.core.credentials import AzureKeyCredential


def setup_client() -> FaceClient:
    """Sets up and returns an Azure FaceClient.

    This function loads environment variables from a `.env` file using
    `load_dotenv()`. It then retrieves the AI service key and endpoint,
    and uses them to authenticate an Azure FaceClient.

    Returns:
        FaceClient: An authenticated instance of the Azure FaceClient.
    """
    load_dotenv()
    key = os.getenv("AI_SERVICE_KEY")
    endpoint = os.getenv("AI_SERVICE_ENDPOINT")

    # Setup face client
    client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return client


def get_face_attributes() -> List[FaceAttributeTypeDetection03]:
    """Returns a list of face attributes to be detected.

    Currently includes:
      - HEAD_POSE
      - BLUR
      - MASK

    Returns:
        List[FaceAttributeTypeDetection03]: A list of face attribute
        enumeration values.
    """
    return [
        FaceAttributeTypeDetection03.HEAD_POSE,
        FaceAttributeTypeDetection03.BLUR,
        FaceAttributeTypeDetection03.MASK,
    ]


def read_image(image_filename: str) -> bytes:
    """Reads an image file in binary mode.

    Args:
        image_filename (str): The path to the image file.

    Returns:
        bytes: The binary content of the image.
    """
    with open(image_filename, mode="rb") as image_data:
        image_content = image_data.read()
    return image_content


def call_face_api(
    client: FaceClient,
    face_attributes: List[FaceAttributeTypeDetection03],
    image_content: bytes,
) -> List[FaceDetectionResult]:
    """
    Calls the Azure Face API to detect faces in the given image content.

    Args:
        client (FaceClient): An authenticated Azure FaceClient instance.
        face_attributes (List[FaceAttributeTypeDetection03]): The face attributes
            to retrieve.
        image_content (bytes): The binary content of the image.

    Returns:
        List[FaceDetectionResult]: A list of detected face objects returned by the API.
    """
    return client.detect(
        image_content=image_content,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=False,
        return_face_attributes=face_attributes,
    )


def show_results(face_results: List[FaceDetectionResult], image_file: str) -> None:
    """Displays and saves the results of the face detection, including bounding boxes.

    This function:
      1. Prints the number of detected faces.
      2. Opens the image with Pillow and draws bounding boxes around each detected face.
      3. Annotates each face with its number.
      4. Prints details about each face's attributes.
      5. Saves the annotated image as 'detected_faces.jpg'.

    Args:
        face_results (List[FaceDetectionResult]): The detected face objects.
        image_file (str): The path to the original image file.
    """
    if len(face_results) > 0:
        print(f"{len(face_results)} faces detected.")

    # Prepare image for drawing
    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = "lightgreen"

    face_count = 0

    # Draw and annotate each face
    for face in face_results:
        face_count += 1
        print(f"\nFace number {face_count}")
        print(f" - Head Pose (Yaw): {face.face_attributes.head_pose.yaw}")
        print(f" - Head Pose (Pitch): {face.face_attributes.head_pose.pitch}")
        print(f" - Head Pose (Roll): {face.face_attributes.head_pose.roll}")
        print(f" - Blur: {face.face_attributes.blur.blur_level}")
        print(f" - Mask: {face.face_attributes.mask.type}")
        print(f" - Nose and mouth covered: {face.face_attributes.mask.nose_and_mouth_covered}")

        # Draw bounding box
        r = face.face_rectangle
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        draw.rectangle(bounding_box, outline=color, width=1)

        annotation = f"Face number {face_count}"
        plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

    # Display and save annotated image
    plt.imshow(image)
    output_file = "faces_detected.jpg"
    fig.savefig(output_file)
    print(f"\nResults saved in {output_file}")


def main() -> None:
    """The main entry point for this script.

    This function:
      1. Sets up the Azure FaceClient.
      2. Retrieves the face attributes to detect.
      3. Reads the image file.
      4. Calls the Azure Face API to detect faces.
      5. Shows and saves the results.
    """
    client = setup_client()
    face_attributes = get_face_attributes()

    image_filename = 'people.jpg'

    image_content = read_image(image_filename=image_filename)
    results = call_face_api(
        client=client,
        face_attributes=face_attributes,
        image_content=image_content,
    )
    show_results(face_results=results, image_file=image_filename)


if __name__ == "__main__":
    main()
