# Face Detection with Azure Cognitive Services

## Overview
This repository contains a python script demoinstrating how to use the [Azure Face Client SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-vision-face-readme?view=azure-python-preview) to detect faces in an image, print face attributes (like head pose, blur, and mask status), and annotate them with bounding boxes. The annotated image is then saved in an image `faces_detected.jpg` for further review.

## Setup

In a `.env` file at the root of the project place the API key and endpoint found in the 'Keys & Endpoint' section within your Azure AI Services resource, like so:

```
AI_SERVICE_ENDPOINT=your_endpoint
AI_SERVICE_KEY=your_key
```
