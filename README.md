# AMB Dashboard

### Create a Conda Environment
1. Create a Conda Environment using the provided `environment.yml` with the command `conda env create --file dashboard_env.yml`
2. Activate the environment by running `conda activate dashboard`
3. Verify that the packages were installed correctly by running `conda list` to see a list of the packages in the environment.

### Running Dashboard
1. Make sure you navigate to the folder where `app.py` is located first
2. Start the dashboard by running the command `python app.py result`. Here, `result` is the folder where all the outputs are stored. The result folder is located in `sim_output/result`
3. Now you can view the dashboard by navigating to `http://127.0.0.1:8050`

### Important
You need to have `assets/individual_diversity_new_infections_travel_prob_revised.json` and `assets/unshifted/` to plot the graphs in the "Spatial Temporal Patterns" tab. It is not uploaded to this repository due to the large file size. 