import os
import glob

def list_files_with_extension(directory, extension):
    search_pattern = os.path.join(directory, f'*.{extension}')
    files = glob.glob(search_pattern)
    return files

#modify this based on your machine
process_num = [4]
epsilon = [0.4, 0.2, 0.05, 0.001]
directory_path = '../inputs/spai/'
extension = 'mtx'
mtx_files = list_files_with_extension(directory_path, extension)
build_dir = "../build/"

os.system(f'cd {build_dir} && cmake .. && make -j 8')

for f in mtx_files:
    print(f)

for p in process_num:
    for mat in mtx_files:
        for eps in epsilon:
            print(f'\nLaunching MPI with {p} process. Problem name: {mat}')
            cmd = f'cd {build_dir} && mpirun -n {p} gmres_test {mat} {eps} 500'
            print("command: ", cmd)
            if os.system(cmd) != 0:
                print(f'Failure: MPI with {p} process. Problem size: {mat}')
            else:
                print(f'Success: MPI with {p} process. Problem name: {mat}\n')