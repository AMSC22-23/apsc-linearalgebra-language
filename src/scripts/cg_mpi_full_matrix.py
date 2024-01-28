import os

#modify this based on your machine
process_num = [1, 2, 4]
input_sizes = [100, 200, 300, 400, 500, 700, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

build_dir = "../build/"

print("Removing old log files...")
os.system(f'cd {build_dir} && rm objective2*.log')
os.system(f'cd {build_dir} && cmake .. && make -j 4')

for p in process_num:
    for i in input_sizes:
        print(f'\nLaunching MPI with {p} process. Problem size: {i}')
        if os.system(f'cd {build_dir} && mpirun -n {p} objective2 {i}') != 0:
            print(f'Failure: MPI with {p} process. Problem size: {i}')
        else:
            print(f'Success: MPI with {p} process. Problem size: {i}\n')
