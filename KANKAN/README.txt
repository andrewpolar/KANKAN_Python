KANKAN - Kolmogorov-Arnold Networks Kaczmarz-Newton

Concept: Andrew Polar, Mike Poluektov
Theory and algorithms are described in peer-reviewed publications.
https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
https://link.springer.com/article/10.1007/s10994-025-06800-6
https://www.mdpi.com/2673-3951/6/3/88
https://www.sciencedirect.com/science/article/pii/S0925231226021703
https://arxiv.org/abs/2609.23906
Additional information, examples and documentation are available at
OpenKAN.org.

The examples can be found in folder /examples

### Build on Windows
**1. Start the build environment.**
Open **x64 Native Tools Command Prompt for VS** (Visual Studio 2026 Build Tools).

**2. Navigate to the KANKAN project folder.**
Execute:
cd /d C:\Users\YourName\Python\KANKAN
Replace `YourName` with your Windows username. You should now see `KANKAN>` at the end of the command prompt.

**3. Find the pybind11 CMake directory.**
Execute:
python -m pybind11 --cmakedir
Copy the full directory path printed by this command.

**4. Configure the project.**
Execute the following command, replacing `PASTE_PATH_HERE` with the path obtained in step 3:
cmake -S . -B build -Dpybind11_DIR="PASTE_PATH_HERE"

**5. Build the Python extension.**
cmake --build build --config Release

**6. Copy the compiled extension into the Python package.**
copy /Y build\Release\_kankan.cp310-win_amd64.pyd python\kankan\

The build is complete. You can now run the Python examples.