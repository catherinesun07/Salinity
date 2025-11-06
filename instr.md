
# virtual environment:
First check if virtualenv is installed:
   ```
   pip show virtualenv

   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
If not installed, you can install it using pip:
   ```
   pip install virtualenv
   ```
Or create a virtual environment using the built-in venv module:
   ```
   python -m venv .venv
   ```  
To activate it in powershell, run:
   ```
   .\.venv\Scripts\Activate.ps1
   ```
To activate in in cmd, run:
   ```
   .\.venv\Scripts\Activate.bat
   ```

 Activate the virtual environment by running the following command in your terminal:
 On Windows:
     ```
     .\.venv\Scripts\Activate
     ```

#Installing packages:
 To install packages within the activated virtual environment, use pip:
   ```
   pip install package_name
   ```
To see installed packages and their versions, use:
   ```
   pip list
   ```
#Running files
To run python scripts within the virtual environment, use:
   ```
   python script_name.py
   ```

Run test function 
    ```
    .\.venv\Scripts\python.exe test_salinity.py
    ```
