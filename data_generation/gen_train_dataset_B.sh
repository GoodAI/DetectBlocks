if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || ! [ -d "$1" ] || ! [ -d "$2" ] || ! [ -d "$3" ]
then
  echo Please specify the following paths in order: path to dataset dir, path to BlenderProc repo, path to output dir. All dirs must exist!
else
  python3 generate_random_views.py --dataset_path "$1" --model_list_file model_subset_B.txt --views_per_model 1600 --output_path poses_dataset_B.txt
  PYOPENGL_PLATFORM='egl' python3 generate_data_opengl.py --model_pose_list poses_dataset_B.txt --output_dir "$3"
  python3 generate_blenderproc_script.py --model_pose_list poses_dataset_B.txt --blenderproc_root "$2" --output_dir "$3"
  source "$3/blenderproc/generate_data_blenderproc.sh"
fi
