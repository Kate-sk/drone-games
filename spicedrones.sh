#!/bin/bash

num=6
model="iris"
world="formation.world"

if [ "$1" == "prof" ]; then
  arg="--gazebo_ros"
elif [ "$1" == "nonprof" ]; then
  arg=""
else
  echo "$0 (prof | nonprof) [mc | vtol] [num] [world]"
  exit
fi

if [ "$2" == "vtol" ]; then
  model="standard_vtol"
fi

if [ "$3" != "" ]; then
  num=$3
fi

if [ "$4" != "" ]; then
  world=$4
fi

./start.sh $model $num gazebo/worlds/$world --ref_point 0,-72,0.1 $arg $@

if [ "$1" == "prof" ]; then
  ./bin/formations_gen.py $model $num formation/borders.txt formation/test_fs/ --names T E C T &
fi

echo "waiting ..."
sleep 3

./examples/spicedrones.py $mode $num

read -p "Press enter to stop ..."

./stop.sh
