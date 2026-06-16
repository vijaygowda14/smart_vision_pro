from django.shortcuts import render

import cv2
import numpy as np

from ultralytics import YOLO

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")


def live_camera_page(request):

    return render(
        request,
        "live_camera.html"
    )


@csrf_exempt

@require_POST

def detect_objects(request):

    image = request.FILES.get('image')

    if not image:
        return JsonResponse({
            'detected_objects': []
        })

    file_bytes = np.asarray(
        bytearray(image.read()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    height, width, _ = frame.shape

    results = model(frame)

    detected = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (x1 + x2) // 2

            object_width = x2 - x1

            # POSITION

            if center_x < width / 3:

                position = "left"

            elif center_x < 2 * width / 3:

                position = "center"

            else:

                position = "right"

            # DISTANCE

            if object_width > 400:

                distance = "danger"

            elif object_width > 250:

                distance = "very close"

            elif object_width > 150:

                distance = "close"

            else:

                distance = "far"

            detected.append({

                "label": label,

                "position": position,

                "distance": distance

            })

    return JsonResponse({

        "detected_objects": detected

    })