
configname="perfect_1car"
python3 ./standalone_project/full_project/testbench.py --algo Dempster --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
python3 ./standalone_project/full_project/testbench.py --algo Conjunctive --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
python3 ./standalone_project/full_project/testbench.py --algo Disjunctive --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname

configname="noisy_1car"
python3 ./standalone_project/full_project/testbench.py --algo Dempster --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
python3 ./standalone_project/full_project/testbench.py --algo Conjunctive --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
python3 ./standalone_project/full_project/testbench.py --algo Disjunctive --save_img True --mean True --json_path ./standalone_project/full_project/configs/config_$configname.json --save_path ~/Desktop/Output_Algo/$configname
