# virtual environment:
First check if virtualenv is installed:
   ```
   pip show virtualenv
   ```
If not installed, you can install it using pip:
   ```
   pip install virtualenv
   ```
Or create a virtual environment using the built-in venv module:
   ```
   python -m venv venv
   ```  
> **Important:** You must run the command above to create the virtual environment before you can activate it.

 To activate it in powershell, run:
   ```
   .\venv\Scripts\Activate.ps1
   ```
> **Note:** If you get an "execution policy" error in PowerShell, see [The PowerShell Fix](#the-powershell-fix) below.

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

### The PowerShell Fix
If you see an error about "execution policy" when trying to activate the virtual environment in PowerShell, it's because of a security setting. You can fix it by running the following command in a PowerShell terminal that you've **opened as an administrator**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
After running this once, you should be able to activate the environment in any PowerShell terminal without issues.

Run test function 
    ```
    .\.venv\Scripts\python.exe test_salinity.py
    ```