# Config file name after "config_" with no extention 
configname="perfect_full_testBBA16"

# Run for each algorithm
echo "Dempster"
python3 ./standalone_project/full_project/testbench.py --algo Dempster --save_img True --mean True --loopback_evid True --pdilate 2 --cooplvl 2 --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname --dataset_path /home/caillot/Documents/Dataset/CARLA_Dataset_original
echo "Conjunctive"
python3 ./standalone_project/full_project/testbench.py --algo Conjunctive --save_img False --mean False --loopback_evid True --pdilate 2 --cooplvl 2 --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname --dataset_path /home/caillot/Documents/Dataset/CARLA_Dataset_original
echo "Disjonctive"
python3 ./standalone_project/full_project/testbench.py --algo Disjunctive --save_img False --mean False --loopback_evid True --pdilate 2 --cooplvl 2 --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname --dataset_path /home/caillot/Documents/Dataset/CARLA_Dataset_original
