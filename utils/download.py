"""
Script to download unmerged outputs from hyperloop
"""

import os
import argparse
import glob
import math
import multiprocessing
import shutil
import subprocess
from pathlib import Path
import uproot

def get_files_from_directory(input_directory, is_slim):
    """
    Function to get all the files from a specific directory
    """

    train_id = input_directory.split(sep="/")[-1]
    if is_slim:
        find_args = ["AnalysisResults.root", "-r"]
    else:
        find_args = ["AOD/*/AnalysisResults.root"]
    with open(f"outputs_{train_id}.txt", "w", encoding="utf-8") as out:
        subprocess.run(["alien_find", f"alien://{input_directory}", *find_args], stdout=out, check=False)

    lines = []
    check_unmerged = False
    with open(f"outputs_{train_id}.txt", encoding="utf-8") as file:
        lines = [(line.rstrip(), train_id) for line in file]

        if len(lines) == 0:
            check_unmerged = True

    if check_unmerged:
        with open(f"outputs_{train_id}.txt", "w", encoding="utf-8") as out:
            subprocess.run(["alien_find", f"alien://{input_directory}", "*/AnalysisResults.root"],
                           stdout=out, check=False)

        with open(f"outputs_{train_id}.txt", encoding="utf-8") as file:
            for line in file:
                if "Stage" in line: # we do not want partially merged files
                    continue
                lines.append((line.rstrip(), train_id))

    if not os.path.isdir(train_id):
        os.mkdir(train_id)

    return lines


def download_file(task_id, file, train_id, runs, an_res_only):
    """
    Function to be executed in parallel that downloads all the files for a specific run
    """

    print(f"Processing task {task_id} - START")
    subprocess.run(["alien_cp", file, f"file:{train_id}/AnalysisResults_{task_id:03d}.root"], check=False)
    # Only select files which have the wanted run numbers
    if runs:
        with uproot.open(f"{train_id}/AnalysisResults_{task_id:03d}.root") as f:
            run_numbers = f["eventselection-run3"]["luminosity"]["hCounterTVX"].axis().labels()
            run_numbers = [int(run) for run in run_numbers if run.isdigit()]
            # check if any of the run numbers is in the list
            if not any(run in runs for run in run_numbers):
                Path(f"{train_id}/AnalysisResults_{task_id:03d}.root").unlink(missing_ok=True)
                return

    if not an_res_only:
        ao2d_cp = subprocess.run(["alien_cp", file.replace('AnalysisResults', 'AO2D'),
                                  f"file:{train_id}/AO2D_{task_id:03d}.root"], check=False)
        if ao2d_cp.returncode != 0:
            # AO2D not found, let's delete also AnalysisResults.root
            Path(f"{train_id}/AnalysisResults_{task_id:03d}.root").unlink(missing_ok=True)

    print(f"Processing task {task_id} - DONE")


def download_and_merge(input_file, num_workers, suffix, n_merged_files, is_slim, runs, an_res_only):  # pylint: disable=too-many-branches
    """
    Main function to download and merge output files from hyperloop
    """

    output_directories = []
    with open(input_file, 'r', encoding="utf-8") as file:
        for line in file:
            # Split line by comma or any delimiter
            parts = line.strip().split(',')
            for part in parts:
                output_directories.append(part)

    num_workers = min(num_workers, os.cpu_count())  # Get the number of available CPU cores)

    # Get the runs from the runs file
    if runs is not None:
        with open(runs, 'r', encoding="utf-8") as file:
            runs = file.read().strip()
        runs = runs.split(',')
        runs = [run.strip() for run in runs if run.strip().isdigit()]
        runs = [int(run) for run in runs]

    with multiprocessing.Pool(processes=num_workers) as pool:
        files = pool.starmap(
            get_files_from_directory,
            [(directory, is_slim) for directory in output_directories]
        )

    files = [item for sublist in files for item in sublist]  # Flatten the list of lists

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            download_file,
            [(i, file[0], file[1], runs, an_res_only) for i, file in enumerate(files)]
        )

    merge_tag = "AnalysisResults" if an_res_only else "AO2D"
    files_to_merge = []
    for train_dir in os.listdir("."):
        if os.path.isdir(train_dir) and "hy_" in train_dir:
            for file in os.listdir(train_dir):
                if merge_tag in file:
                    file_path = os.path.join(train_dir, file)
                    files_to_merge.append(file_path)

    n_files_perbunch = math.ceil(len(files_to_merge) / n_merged_files)
    nbunch = 0
    for ifile, file in enumerate(files_to_merge):
        if ifile % n_files_perbunch == 0:
            nbunch += 1
        with open(f"files_to_merge_{nbunch}.txt", "a", encoding="utf-8") as merge_list:
            merge_list.write(f"{file}\n")

    if not an_res_only:
        if nbunch > 1:
            for bunch in range(1, nbunch+1):
                subprocess.run(["o2-aod-merger", "--input", f"files_to_merge_{bunch}.txt",
                                "--output", f"AO2D{suffix}_{bunch}.root",
                                "--max-size", "100000000", "--skip-parent-files-list"], check=False)
        else:
            subprocess.run(["o2-aod-merger", "--input", "files_to_merge_1.txt",
                            "--output", f"AO2D{suffix}.root",
                            "--max-size", "100000000", "--skip-parent-files-list"], check=False)

    if an_res_only and nbunch > 1:
        for bunch in range(1, nbunch + 1):
            subprocess.run(["hadd", "-f", f"AnalysisResults{suffix}_{bunch}.root",
                            f"@files_to_merge_{bunch}.txt"], check=False)
    else:
        subprocess.run(["hadd", "-f", f"AnalysisResults{suffix}.root",
                        *sorted(glob.glob("hy_*/AnalysisResults*.root"))], check=False)

    for train_dir in glob.glob("hy_*"):
        if os.path.isdir(train_dir):
            shutil.rmtree(train_dir)
    for tmp_file in glob.glob("outputs_hy_*") + glob.glob("files_to_merge_*"):
        os.remove(tmp_file)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Arguments")
    PARSER.add_argument("--input_file", "-i", metavar="text",
                        help="text input file with directories", required=True)
    PARSER.add_argument("--jobs", "-j", type=int, default=20,
                        help="number of workers", required=False)
    PARSER.add_argument("--suffix", "-s", metavar="text",
                        default="_LHC24_pass1_skimmed",
                        help="output file", required=False)
    PARSER.add_argument("--n_merged_files", "-n", type=int,
                        default=1, help="number of output merged AO2D files",
                        required=False)
    PARSER.add_argument("--slim", action="store_true", default=False,
                        help="option for slim outputs on hyperloop", required=False)
    PARSER.add_argument("--runs", metavar="text", required=False, default=None,
                        help="text input file with run numbers to download (only for non-slim files)")
    PARSER.add_argument("--an_res_only", action="store_true", default=False,
                        help="option for only downloading analysis results (not AO2D)", required=False)
    ARGS = PARSER.parse_args()

    download_and_merge(
        ARGS.input_file, ARGS.jobs, ARGS.suffix, ARGS.n_merged_files,
        ARGS.slim, ARGS.runs, ARGS.an_res_only
    )
