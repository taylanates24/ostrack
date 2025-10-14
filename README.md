# ostrack

## Install the environment


**1 - Clone the repository and navigate to the project folder**

```
git clone https://github.com/taylanates24/ostrack.git 
```

```
cd ostrack
```

**2- Install from the Dcoekrfile**

```
docker build -t ostrack -f Dockerfile .
```


### Set project paths

Run the following command to set paths for this project
```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files

`lib/train/admin/local.py`  # paths about training
`lib/test/evaluation/local.py`  # paths about testing


## How to run video_demo.py

**1. Download Pre-trained Models**

Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd).

Put the downloaded `models` directory in the project's root directory.

**2. Run the docker container**

First, you need to run the Docker container you built. The following command will start the container, mount your current project directory into the container, and open a bash shell.

```bash
docker run -it --rm --gpus all -v $(pwd):/ostrack -w /ostrack ostrack bash
```
*Note: We add `--gpus all` to enable GPU support. If you don't have a GPU, you can remove `--gpus all`.*
*If you are on Windows, you might need to replace `$(pwd)` with the absolute path to the project directory.*

**3. Run the demo script**

Once you are inside the container's shell, you can run the `video_demo.py` script.

To run with a specific video file:
```bash
python tracking/video_demo.py --videofile /path/to/your/video.mp4
```

To run with the default video provided in the repository (`videos/1_ir.mp4`):
```bash
python tracking/video_demo.py
```

You can see all the available options by running:
```bash
python tracking/video_demo.py --help
```


