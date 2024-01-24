import os
import glob
import csv
from plot_objective3 import plot

def list_files_with_extension(directory, extension):
    search_pattern = os.path.join(directory, f'objective3*.{extension}')
    files = glob.glob(search_pattern)
    return files

directory_path = './i7-12700k__3600_MHz_DDR4/'
extension = 'log'

log_files = list_files_with_extension(directory_path, extension)
problem_dict = {f'{f}': {} for f in log_files}
solver_sizes = [f'{f}' for f in log_files]
problem_names = []

# Build the global dictionary that maps mpi size to sparse problem
for i, f in enumerate(log_files):
    with open(f, 'r') as csv_file:
        reader = csv.reader(csv_file)
        first = True
        for row in reader:
            # Skip csv headers
            if first == True:
                first = False
                continue
            problem_dict[f'{f}'][f'{row[0]}'] = row[2]
            # If first iteration save all the problem names
            if i == 0:
                problem_names.append(row[0])

prefix = f'{directory_path}objective3_MPISIZE'
for problem in problem_names:
    # Enhance this :)
    # Recover the data regarding the current solver and problem name
    solver_map = {((solver[len(prefix):])[:-4]): int(problem_dict[solver][problem]) for solver in solver_sizes}
    plot(solver_map, directory_path + problem)
