OS: Ubuntu 24.04

```bash
apt update
apt install -y \
		git \
		libeigen3-dev \
		cmake \
		build-essential \
		patchelf
ROOT_PATH=/path/to/ROOT_PATH
mkdir -p $ROOT_PATH
```
Prepare the environment and install dependencies
```bash
cd $ROOT_PATH
git clone https://github.com/intel-sandbox/JiehongLin-SAM-6D.git sam6d
cd sam6d
git checkout pravin-dev
cd SAM-6D
conda env create -f ov_environment_u24.yaml
conda activate ov_sam6d
```
Prepare the openvino with fixes
```bash
cd $ROOT_PATH
git clone https://github.com/pravin25/openvino.git ovArgMaxFix
cd ovArgMaxFix
git checkout arg_max_fix_clean
git submodule update --init --recursive
sudo ./install_build_dependencies.sh
# make sure python from conda environment is used
python -m pip install -r src/bindings/python/wheel/requirements-dev.txt
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_WHEEL=ON -S . -B build --fresh
cmake --build build --parallel $(nproc) --clean-first
python -m pip install --force-reinstall $ROOT_PATH/ovArgMaxFix/build/wheels/openvino-2026.0.0-20939-cp311-cp311-manylinux_2_39_x86_64.whl
```
Compile the ov_extension library for sam6d
```bash
cd $ROOT_PATH/sam6d/SAM-6D/Pose_Estimation_Model/model/ov_pointnet2_op
cmake -S . -B build
cmake --build build
```

Remove existing output for FE & PEM
```bash
rm -r $ROOT_PATH/sam6d/SAM-6D/Data/Example/outputs/fe_debug
rm -r $ROOT_PATH/sam6d/SAM-6D/Data/Example/outputs/pem_debug
```
Run the FE pipeline on CPU/GPU
```bash
python run_fe_golden.py --device CPU
python run_fe_golden.py --device GPU
```

Run the PEM pipeline on CPU and GPU
```bash
python run_pem_golden.py --device CPU
python run_pem_golden.py --device GPU
python run_pem_six_batch_golden.py --device CPU
python run_pem_six_batch_golden.py --device GPU
```
CPU/GPU output saved inside `$ROOT_PATH/sam6d/SAM-6D/Data/Example/outputs/pem_debug/`. Check `vis_pem_ov_CPU.png`, `vis_pem_ov_GPU.png`.


Known issues/pointers to note:
- PEM pipeline is working except Feature Extraction (FE). FE is WIP
- export of PEM model to xml is broken. Test using _golden_ xml files provided in the repo for now. export is WIP.
