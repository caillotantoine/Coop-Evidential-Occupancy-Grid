# Config file name after "config_" with no extention 
configname="perfect_full_testBBA15"

# Run for each algorithm
echo "Dempster"
python3 ./standalone_project/full_project/testbench.py --algo Dempster --save_img False --mean False --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
echo "Conjunctive"
python3 ./standalone_project/full_project/testbench.py --algo Conjunctive --save_img False --mean False --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
echo "Disjonctive"
python3 ./standalone_project/full_project/testbench.py --algo Disjunctive --save_img False --mean False --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
