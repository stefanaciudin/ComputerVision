import os


def change_files_extension_to_png(folder_path):
    if not os.path.isdir(folder_path):
        print(f"The specified folder does not exist: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only rename files (ignore directories)
        if os.path.isfile(file_path):
            # Split the file name and its extension
            name, ext = os.path.splitext(filename)

            # If the file already has a .png extension, skip it
            if ext != ".png":
                # Define the new file name with .png extension
                new_file_path = os.path.join(folder_path, f"{name}.png")
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {name}.png")
            else:
                print(f"Skipped (already .png): {filename}")


change_files_extension_to_png("lab3/b/Pratheepan_Dataset/FamilyPhoto")
