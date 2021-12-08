if [ -z $1 ] || ! [ -d "$1" ]
then
  echo Please specify the path to the dataset directory!
else
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -pink_background --pose_status known
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -pink_background --pose_status perturbed
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -pink_background --pose_status unknown
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 --pose_status known
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 --pose_status perturbed
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 --pose_status unknown
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view -pink_background --pose_status known
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view -pink_background --pose_status perturbed
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view -pink_background --pose_status unknown
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view --pose_status known
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view --pose_status perturbed
  PYOPENGL_PLATFORM='egl' python evaluate_on_dataset.py "$1" --cfg_name ult4 -multi_view --pose_status unknown
fi
