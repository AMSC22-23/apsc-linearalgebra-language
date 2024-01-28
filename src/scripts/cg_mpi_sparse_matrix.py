import os
import glob

def list_files_with_extension(directory, extension):
    search_pattern = os.path.join(directory, f'*.{extension}')
    files = glob.glob(search_pattern)
    return files

#modify this based on your machine
process_num = [1, 2, 4]
directory_path = '../inputs/'
extension = 'mtx'
mtx_files = list_files_with_extension(directory_path, extension)
build_dir = "../build/"

print("Removing old log files...")
os.system(f'cd {build_dir} && rm objective3*.log')
os.system(f'cd {build_dir} && cmake .. && make -j 4')

for f in mtx_files:
    print(f)

for p in process_num:
    for mat in mtx_files:
        print(f'\nLaunching MPI with {p} process. Problem name: {mat}')
        if os.system(f'cd {build_dir} && mpirun -n {p} objective3 {mat}') != 0:
            print(f'Failure: MPI with {p} process. Problem size: {mat}')
        else:
            print(f'Success: MPI with {p} process. Problem name: {mat}\n')
