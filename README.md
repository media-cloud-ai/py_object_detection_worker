# Object Detection Worker
Detection of objects in video using OpenCV Yolo v3.

## Build docker image locally (optional)

```bash
docker build -t mediacloudai/py_object_detection_worker .
```

## Run worker locally

Export variables to setup Source and Destination filenames:
```bash
export SOURCE_FOLDER=replace_with_folder_with_the_source_filename
export SOURCE_PATH=replace_with_input_filename
export DESTINATION_PATH=replace_with_output_filename
```

```bash
docker run --rm \
  -v `pwd`/examples:/examples \
  -e RUST_LOG=debug \
  -e SOURCE_ORDERS=/examples/job_example.json \
  -v $SOURCE_FOLDER:/movies \
  -e SOURCE_PATH=/movies/$SOURCE_PATH \
  -e DESTINATION_PATH=/movies/$DESTINATION_PATH \
  mediacloudai/py_object_detection_worker
```
