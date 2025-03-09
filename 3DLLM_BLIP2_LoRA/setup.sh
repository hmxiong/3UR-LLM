cd lavis/models/blip2_models/EPCL/third_party/pointnet2/
python setup.py install
cd ../../utils/
pip install cython
python cython_compile.py build_ext --inplace

pip install trimesh, plyfile
mim install mmcv-full==1.7.2