# ASL Alphabet Object Detection

A TensorFlow notebook for detecting American Sign Language (ASL) alphabet hand gestures in images. It fine-tunes **SSD MobileNet V2 FPNLite (320 × 320)** on 26 letter classes, **A–Z**, and walks through training, evaluation, and image inference.

This is an educational object detection project. It identifies letter gestures in still images; it does not translate continuous signing or model the motion used in letters such as J and Z.

## Start here

Browse [object_detection.ipynb](object_detection.ipynb) to see the complete workflow and saved outputs, or open it in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrigank-shome/object-detection/blob/main/object_detection.ipynb)

The repository contains the notebook, including outputs from an earlier run. **Training images, annotations, and the custom trained checkpoint are not included.** The notebook needs the data and path adjustments described below before it can be rerun.

## Dataset

The identified source is **American Sign Language Letters v1**, hosted by Roboflow:

- [Original dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1)
- [Public GitHub mirror](https://github.com/paulinamoskwa/Real-Time-Sign-Language/tree/master/American%20Sign%20Language%20%28ASL%29%20dataset)

The mirror contains 1,728 images with bounding box annotations for 26 classes:

| Split | Images |
| --- | ---: |
| Training | 1,512 |
| Validation | 144 |
| Test | 72 |

The notebook's example image, `W22_jpg.rf.b08e3a39e8878bf5cb1cf51638759eed.jpg`, is present in the mirror's validation split and matches the saved notebook output. This identifies the example's source; the exact data split used in the original training run is not documented.

**Annotation format matters:** the mirror uses YOLO text labels, while the notebook's `generate_tfrecord.py` workflow expects Pascal VOC XML. Export Pascal VOC annotations from the original source if available, or convert the mirror's labels before generating TFRecords. YOLO class IDs are zero-based; the notebook's TensorFlow label map uses IDs 1–26.

Keep validation and test data separate from training data. The notebook uses a folder named `test` for evaluation during development; use validation data there and reserve the held-out test split for final reporting.

## Requirements

The workflow uses Python, TensorFlow 2, the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), NumPy, pandas, OpenCV, Matplotlib, and a protobuf compiler. Annotation conversion uses [GenerateTFRecord](https://github.com/nicknochnack/GenerateTFRecord).

A development environment was checked with **Python 3.11, TensorFlow 2.15.1, tf-models-official 2.15.0, protobuf 3.20.3, and NumPy 1.26.4**. The check covered SSD MobileNet construction and inference on a synthetic image, two Object Detection API tests, and Jupyter notebook serving. It did not reproduce full training or validate a custom checkpoint.

There is currently no dependency manifest or lockfile in this repository. The notebook contains older installation cells that upgrade and downgrade packages independently. Treat those cells as historical setup instructions: use a compatible, isolated environment and consult the [Object Detection API installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) before running them. A GPU is recommended for training; the development smoke check ran on CPU.

## Prepare and run

1. Open the notebook in Colab, or use Jupyter in a compatible local environment. The Google Drive mount cell applies to Colab.
2. Download the dataset and prepare paired image/XML files. Put development evaluation data in `images/test` as expected by the notebook.
3. Set `destination` to your project directory. The existing default is `/content/gdrive/MyDrive/final_object_recognition`. Update the other hard-coded Drive paths too, including the API installation command, pretrained inference paths, prediction image path, and TensorBoard paths.
4. Install the Object Detection API and compile its protobuf definitions. Run its installation verification before proceeding.
5. Download the pretrained [SSD MobileNet V2 FPNLite checkpoint](https://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz), then generate the A–Z label map and TFRecords.
6. Configure the pipeline and train. The notebook sets a batch size of 4 and requests 10,000 training steps; adjust these to your available hardware.
7. Evaluate and run image inference using a checkpoint produced by your run. Replace the hard-coded `ckpt-11` with an existing checkpoint and choose your prediction image.

The data layout under `destination` should look like this after preparation:

```text
final_object_recognition/
├── prediction/
│   └── your-image.jpg
└── Tensorflow/
    ├── models/                       # TensorFlow models / Object Detection API
    ├── scripts/
    │   └── generate_tfrecord.py
    └── workspace/
        ├── images/
        │   ├── train/                # Images with matching Pascal VOC XML files
        │   └── test/                 # Development evaluation images and XML files
        ├── annotations/              # Generated label map and TFRecords
        ├── pre-trained-models/       # Downloaded base model
        └── models/
            └── my_ssd_mobnet/        # Pipeline config and training checkpoints
```

## Notebook workflow

| Stage | What it does |
| --- | --- |
| Setup | Mount Drive, define paths, and prepare dependencies |
| Data preparation | Create the label map and convert XML annotations to TFRecords |
| Model configuration | Set the class count, checkpoint, batch size, and input paths |
| Training | Fine-tune SSD MobileNet on the letter dataset |
| Evaluation | Calculate COCO-style detection metrics |
| Inference | Load a checkpoint and draw predicted boxes on an image |
| Export and inspection | Export the model and inspect TensorBoard logs |

## Known limitations

- The notebook relies on external data and a custom checkpoint that are not stored here.
- Paths and checkpoint names are specific to the original Colab session.
- The pretrained COCO comparison currently shares the custom A–Z label map. Use the COCO label map to display pretrained predictions correctly.
- Saved outputs are historical examples, not evidence of a reproducible benchmark or accuracy on new images.

## References

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- [GenerateTFRecord](https://github.com/nicknochnack/GenerateTFRecord)
- [American Sign Language Letters dataset](https://public.roboflow.com/object-detection/american-sign-language-letters/1)
