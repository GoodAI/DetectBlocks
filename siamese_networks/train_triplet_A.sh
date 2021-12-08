if [ -z $1 ] || ! [ -d "$1" ] || [ -z $2 ] || ! [ -d "$2" ]
then
  echo Please specify the path to the directories with OpenGL images and BlenderProc images!
else
  python triplet.py "$1" "$2" --exp_name default --num_models 16 --imgs_per_model 7998
fi
