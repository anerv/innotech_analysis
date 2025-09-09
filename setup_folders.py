# %%
import os

parent_folders = ["data", "results", "illustrations"]

for f in parent_folders:
    if not os.path.exists(f):
        os.mkdir(f)

        print("Successfully created folder: " + f)


data_folders = ["input", "processed"]
for f in data_folders:
    if not os.path.exists(os.path.join("data", f)):
        os.mkdir(os.path.join("data", f))

        print("Successfully created folder: " + f)


data_input_subfolders = ["election"]
for f in data_input_subfolders:
    if not os.path.exists(os.path.join("data", "input", f)):
        os.mkdir(os.path.join("data", "input", f))

        print("Successfully created folder: " + os.path.join("data", "input", f))


result_subfolders = ["data", "maps", "plots"]

for f in result_subfolders:
    if not os.path.exists(os.path.join("results", f)):
        os.mkdir(os.path.join("results", f))

        print("Successfully created folder: " + os.path.join("results", f))

# %%
