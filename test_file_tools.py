"""
Test script for the Folder Filename Builder node
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the node
from categories.file_tools import CYHFolderFilenameBuilderNode

def test_folder_filename_builder():
    """Test the Folder Filename Builder node with various inputs"""
    node = CYHFolderFilenameBuilderNode()
    
    # Test case 1: Default values
    result = node.build_path(
        project_name="MyProject",
        filename="image01",
        use_subfolders=True,
        delimiter="/",
        subfolder="images"
    )
    print(f"Test 1 - Default values: {result[0]}")
    assert result[0] == "MyProject/images/image01", f"Expected 'MyProject/images/image01', got '{result[0]}'"
    
    # Test case 2: No subfolder
    result = node.build_path(
        project_name="MyProject",
        filename="image01",
        use_subfolders=False,
        delimiter="/",
        subfolder="images"
    )
    print(f"Test 2 - No subfolder: {result[0]}")
    assert result[0] == "MyProject/image01", f"Expected 'MyProject/image01', got '{result[0]}'"
    
    # Test case 3: Different delimiter
    result = node.build_path(
        project_name="MyProject",
        filename="image01",
        use_subfolders=True,
        delimiter="_",
        subfolder="images"
    )
    print(f"Test 3 - Underscore delimiter: {result[0]}")
    assert result[0] == "MyProject_images_image01", f"Expected 'MyProject_images_image01', got '{result[0]}'"
    
    # Test case 4: Empty project name
    result = node.build_path(
        project_name="",
        filename="image01",
        use_subfolders=True,
        delimiter="/",
        subfolder="images"
    )
    print(f"Test 4 - Empty project name: {result[0]}")
    assert result[0] == "UntitledProject/images/image01", f"Expected 'UntitledProject/images/image01', got '{result[0]}'"
    
    # Test case 5: Empty filename
    result = node.build_path(
        project_name="MyProject",
        filename="",
        use_subfolders=True,
        delimiter="/",
        subfolder="images"
    )
    print(f"Test 5 - Empty filename: {result[0]}")
    assert result[0] == "MyProject/images/untitled", f"Expected 'MyProject/images/untitled', got '{result[0]}'"
    
    # Test case 6: Invalid characters
    result = node.build_path(
        project_name="My<Project>",
        filename="image<01>",
        use_subfolders=True,
        delimiter="/",
        subfolder="im/ages"
    )
    print(f"Test 6 - Invalid characters: {result[0]}")
    assert result[0] == "MyProject/images/image01", f"Expected 'MyProject/images/image01', got '{result[0]}'"
    
    # Test case 7: Empty subfolder with use_subfolders=True
    result = node.build_path(
        project_name="MyProject",
        filename="image01",
        use_subfolders=True,
        delimiter="/",
        subfolder=""
    )
    print(f"Test 7 - Empty subfolder with use_subfolders=True: {result[0]}")
    assert result[0] == "MyProject/image01", f"Expected 'MyProject/image01', got '{result[0]}'"
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_folder_filename_builder()