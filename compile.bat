gcc -c -g ct_main.c
gcc -c -g ct_backprop.c
gcc -c -g ct_util.c
gcc -c -g ct_layer.c
gcc -c -g main.c

gcc -o main.exe ct_main.o ct_backprop.o ct_util.o ct_layer.o main.o