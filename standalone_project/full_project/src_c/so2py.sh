echo -e "import numpy as np\nfrom ctypes import *\n"
libname=$(echo $1 | cut -d'.' -f1)
echo "$libname = cdll.LoadLibrary('$(pwd $1)/$1')"
for cmd in $(nm -gD --demangle $1 -g | grep ' T ' | grep -v '(' | cut -d' ' -f3-)
do 
    func=$(objdump -CS $1 | grep -v 'main(' | grep $cmd'(')
    if [ -n "$func" ]
    then
        echo -e "\n"
        echo '# '$func
        name=$(echo $func | cut -d' ' -f2 | cut -d'(' -f1)
        restype=$(echo $func | cut -d' ' -f1) 
        param=$(echo $func | cut -d'(' -f2 | cut -d')' -f1)
        # echo $restype
        # echo $name
        # echo $param
        
        if [ "$restype" != "void" ]
        then
            echo "$libname.$name.restype = c_$restype"
        fi
        if [ -n "$param" ]
        then
            echo -n "$libname.$name.argtypes = ["
            for p in $(echo $param | tr " " "_" | tr "," "\n")
            do
                if [ -n "$(echo $p | grep "*")" ]; then
                    echo -n 'np.ctypeslib.ndpointer(dtype=np.'
                    if [ -n "$(echo $p | grep "unsigned")" ]; then
                        echo -n 'u'
                    fi
                    if [ -n "$(echo $p | grep "char")" ]; then
                        echo -n 'int8'
                    fi
                    if [ -n "$(echo $p | grep "int")" ]; then
                        echo -n 'int32'
                    fi
                    if [ -n "$(echo $p | grep "float")" ]; then
                        echo -n 'float32'
                    fi
                    if [ -n "$(echo $p | grep "double")" ]; then
                        echo -n 'float64'
                    fi
                    echo -n ')'
                else
                    if [ -n "$(echo $p | grep "char")" ]; then
                        echo -n 'c_char'
                    fi
                    if [ -n "$(echo $p | grep "int")" ]; then
                        echo -n 'c_int'
                    fi
                    if [ -n "$(echo $p | grep "float")" ]; then
                        echo -n 'c_float'
                    fi
                    if [ -n "$(echo $p | grep "double")" ]; then
                        echo -n 'c_double'
                    fi
                fi
                echo -n ', '
            done
            echo -e '\b\b]'
        fi


        echo -n "def "$name"_w("
        if [ -n "$param" ]
        then
            for p in $(echo $param | tr " " "~" | tr "," "\n")
            do
                varname=$(echo $p | rev | cut -d'~' -f1 | rev | tr '*' ' ')
                echo -n "$varname:"
                if [ -n "$(echo $p | grep "*")" ]; then
                    echo -n 'np.ctypeslib.ndpointer(dtype=np.'
                    if [ -n "$(echo $p | grep "unsigned")" ]; then
                        echo -n 'u'
                    fi
                    if [ -n "$(echo $p | grep "char")" ]; then
                        echo -n 'int8'
                    fi
                    if [ -n "$(echo $p | grep "int")" ]; then
                        echo -n 'int32'
                    fi
                    if [ -n "$(echo $p | grep "float")" ]; then
                        echo -n 'float32'
                    fi
                    if [ -n "$(echo $p | grep "double")" ]; then
                        echo -n 'float64'
                    fi
                    echo -n ')'
                else
                    if [ -n "$(echo $p | grep "char")" ]; then
                        echo -n 'c_char'
                    fi
                    if [ -n "$(echo $p | grep "int")" ]; then
                        echo -n 'c_int'
                    fi
                    if [ -n "$(echo $p | grep "float")" ]; then
                        echo -n 'c_float'
                    fi
                    if [ -n "$(echo $p | grep "double")" ]; then
                        echo -n 'c_double'
                    fi
                fi
                echo -n ', '
            done 
            echo -e -n '\b):\n\t'
        else
            echo -e -n '):\n\t'
        fi
        
        echo -n "$libname.$name("
        if [ -n "$param" ]
        then
            for p in $(echo $param | tr " " "~" | tr "," "\n")
            do
                varname=$(echo $p | rev | cut -d'~' -f1 | rev | tr '*' ' ')
                echo -n "$varname, "
            done
            echo -e -n '\b\b)\n'
        else
            echo ')'
        fi
    fi
done