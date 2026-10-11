from pathlib import Path

def select_png(directory):
    """
    Let the user select a PNG file from a directory.

    Returns:
        Path: the selected PNG file
    Raises:
        ValueError: if the directory is invalid or contains no PNG files
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Not a valid directory: {directory}")

    # Only include regular files ending in .png (case-insensitive)
    png_files = sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    if not png_files:
        raise ValueError(f"No PNG files found in: {dir_path}")

    print("Select a PNG file:")

    for i, png_file in enumerate(png_files, start=1):
        print(f"{i}. {png_file.name}")

    while True:
        choice = input("Enter a number: ").strip()

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(png_files):
                selected = png_files[index - 1]
                print(f"Selected: {selected}")
                return selected

        print("Invalid selection. Please enter a number from the list.")


def pick_from_list(input_list: list, info_str: str = ""):

    print(info_str)

    index = 0
    for item in input_list:
        print(f"{index}: {item}")
        index +=1

    user_choice = input("select a number from above list: ")
    user_choice = int(user_choice)

    list_selection = input_list[user_choice]

    return list_selection




# Example usage
if __name__ == "__main__":
    image = select_png("path/to/your/images")
    print(f"Loading: {image}")