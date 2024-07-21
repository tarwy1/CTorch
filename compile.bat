gcc -c -g ct_network.c
gcc -c -g ct_optimizer.c
gcc -c -g ct_util.c
gcc -c -g ct_layer.c
gcc -c -g main.c

gcc -o main.exe ct_network.o ct_optimizer.o ct_util.o ct_layer.o main.o