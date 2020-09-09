import glob
import os
ipy_list = glob.glob("*.ipynb")
sh_name = "check.sh"
kernel_name = "pixyz_py38"
command_ = f"jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name={kernel_name} --execute --inplace "



if os.path.exists(sh_name):
    os.remove(sh_name)
for ipy in ipy_list:
    with open("check.sh", "a") as f:
        f.write(command_ + ipy + "\n")