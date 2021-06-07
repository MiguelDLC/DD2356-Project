The code should simply be compiled with

cc -o main.ou main.c
or 
mpicc -o main.ou main.c

although adding -O3 is highly recommended.

Runing the code will produce traces for each process containing the timigs of the simulation. The parameters can be changed at the beginning of the main function.
