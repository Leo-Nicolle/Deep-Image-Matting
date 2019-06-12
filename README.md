# Deep Image Matting

image matting from [this repo](https://github.com/foamliu/Deep-Image-Matting) just cleaned from the training stuff

## Dependencies

- install deps with

```
pip install -r requirements.txt
```

- Download the [model](https://github.com/foamliu/Deep-Image-Matting/releases/download/v1.0/final.42-0.0398.hdf5) and
  place it in _models/_ folder

## Usage

input, trimap and output should be 320x320 pixels

```
python main image trimap out
```
