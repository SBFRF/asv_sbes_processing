# Contains function to make a mission summary YAML and a function to make a mission failure report YAML (in progress)

import yaml
import os


def _is_non_interactive():
    """Check if running in non-interactive mode.

    Returns True if:
    - SWACSS_NON_INTERACTIVE env var is set to '1', 'true', or 'yes'
    - MPLBACKEND env var contains 'agg' (case insensitive)
    """
    if os.environ.get('SWACSS_NON_INTERACTIVE', '').lower() in ('1', 'true', 'yes'):
        return True
    if 'agg' in os.environ.get('MPLBACKEND', '').lower():
        return True
    return False


# Mission Summary YAML File
def make_summary_yaml(datadir):

    # Create yaml file name
    mission_summary_fname = r"mission_summary_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, mission_summary_fname)

    non_interactive = _is_non_interactive()

    # First, check to see if file already exists; if so, decide if you want to overwrite it
    if os.path.exists(file_path):
        if non_interactive:
            # In non-interactive mode, always overwrite
            print(f"{mission_summary_fname} exists. Non-interactive mode: overwriting.")
        else:
            while True:
                overwrite = input(mission_summary_fname + " already exists. Do you want to overwrite it? (y/n): ")
                try:
                    if overwrite.strip().lower() != "y" and overwrite.strip().lower() != "n":
                        print("User input must be 'y' or 'n'. Try again.")
                    else:
                        break
                except ValueError:
                    print("Input invalid. Please enter 'y' or 'n'.")

            if overwrite.strip().lower() == "n":
                print("File will not be overwritten. Exiting function.")
                return
            if overwrite.strip().lower() == "y":
                print("File will be overwritten.")


    # Initialize a dictionary with four pre-defined keys and empty values
    mission_metadata = {
        "frf": "",
        "overlap": "",
        "repeat": "",
        "repeat_count": ""
    }

    mission_questions = [
        "Is this an FRF mission? (y/n): ",
        "Did you overlap any FRF survey lines? (y/n): ",
        "Did you repeat lines? (y/n): ",
        "How many times did you repeat lines? (enter a number): "
    ]

    if non_interactive:
        # Use default values for automated testing
        # Default: non-FRF mission, no overlap, no repeats
        mission_metadata = {
            "frf": "n",
            "overlap": "n",
            "repeat": "n",
            "repeat_count": ""
        }
        user_notes = "\n" + "# User Notes: Generated in non-interactive mode"
        print("Non-interactive mode: using default mission summary values (frf=n, overlap=n, repeat=n)")
    else:
        # Prompt the user to enter a value for each key, organize into dictionary object
        print('## Populate mission summary YAML file by answering the following questions:')
        for i, key in enumerate(mission_metadata.keys()):
            if i == 3 and mission_metadata["repeat"].strip().lower() == "n":
                # Skip last question if answer to "did you repeat" is "n"
                mission_metadata[key] = ""
                break
            if i < 3:
                while True:
                    mission_metadata[key] = input(mission_questions[i]).strip().lower()
                    try:
                        if mission_metadata[key].strip().lower() != "y" and mission_metadata[key].strip().lower() != "n":
                            print("User input must be 'y' or 'n'. Try again.")
                        else:
                            break
                    except ValueError:
                        print("Input invalid. Please enter 'y' or 'n'.")
            else:
                while True:
                    mission_metadata[key] = input(mission_questions[i]).strip().lower()
                    try:
                        if not mission_metadata[key].isdigit():
                            print("User input must be an integer. Try again.")
                        else:
                            break
                    except ValueError:
                        print("Input invalid. Please enter an integer.")


        # Prompt the user to enter any additional notes
        user_notes_prompt = input("Optional -- Enter any additional notes (hit enter to continue): ")
        if not user_notes_prompt:
            user_notes = "\n" + "# User Notes: None"
        else:
            user_notes = "\n" + "# User Notes: " + user_notes_prompt

    # Define the preamble text
    preamble = "# This is a YAML file containing a dictionary of user responses to the following questions about the mission:\n" + "\n".join(f"#     {item}" for item in mission_questions) + "\n"


    # Write the dictionary to YAML file
    with open(file_path, "w") as file:
        # Write the preamble text
        file.write(preamble)
        # Write the dictionary as YAML
        yaml.dump(mission_metadata, file)
        # Write the optional notes
        file.write(user_notes)
    print("Responses written to ", mission_summary_fname, " in mission directory.")


########################################################################################################################


# # Mission Failure YAML File
def make_failure_yaml(datadir):

    # Create yaml file name
    failure_fname = r"mission_failure_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, failure_fname)

    non_interactive = _is_non_interactive()

    # First, check to see if file already exists; if so, decide if you want to overwrite it
    if os.path.exists(file_path):
        if non_interactive:
            # In non-interactive mode, always overwrite
            print(f"{failure_fname} exists. Non-interactive mode: overwriting.")
        else:
            while True:
                overwrite = input(failure_fname + " already exists. Do you want to overwrite it? (y/n): ")
                try:
                    if overwrite.strip().lower() != "y" and overwrite.strip().lower() != "n":
                        print("User input must be 'y' or 'n'. Try again.")
                    else:
                        break
                except ValueError:
                    print("Input invalid. Please enter 'y' or 'n'.")

            if overwrite.strip().lower() == "n":
                print("File will not be overwritten. Exiting function.")
                return
            if overwrite.strip().lower() == "y":
                print("File will be overwritten.")


    # Initialize a dictionary with four pre-defined keys and empty values
    failure_metadata = {
        "mission_failure": "",
        "mechanical_failure": "",
        "data_quality_failure": "",
        "condition_failure": "",
    }

    failure_questions = [
        "Was there any kind of mission failure? (y/n): ",
        "Enter category of mechanical failure -- \n   0=no mechanical failure, \n   1=something broke but mission was still accomplished, \n   2=vehicle rescue required: ",
        "Enter category of quality failure -- \n   0=no data failure, \n   1=data failure: ",
        "Enter category of condition failure -- \n   0=no conditions failure, \n   1=comms, \n   2=hydrodynamic (i.e. breakpoint), \n   3=environmental (i.e. biofouling): "
    ]

    if non_interactive:
        # Use default values for automated testing
        # Default: no mission failure
        failure_metadata = {
            "mission_failure": "n",
            "mechanical_failure": "",
            "data_quality_failure": "",
            "condition_failure": "",
        }
        failure_comments = ""
        user_notes = "\n" + "# Additional User Notes: Generated in non-interactive mode"
        print("Non-interactive mode: using default failure values (mission_failure=n)")
    else:
        # Prompt the user to enter a value for each key, organize into dictionary object
        print('## Populate mission failure YAML file by answering the following questions:')

        failure_comments = ""     # Create empty notes string
        for i, key in enumerate(failure_metadata.keys()):
            if i > 0 and failure_metadata["mission_failure"].strip().lower() == "n":
                break
            if i == 0:
                while True:
                    failure_metadata[key] = input(failure_questions[i]).strip().lower()
                    try:
                        if failure_metadata[key].strip().lower() != "y" and failure_metadata[key].strip().lower() != "n":
                            print("User input must be 'y' or 'n'. Try again.")
                        else:
                            break
                    except ValueError:
                        print("Input invalid. Please enter 'y' or 'n'.")
            else:
                while True:
                    failure_metadata[key] = input(failure_questions[i]).strip().lower()
                    try:
                        if not failure_metadata[key].isdigit():
                            print("User input must be an integer. Try again.")
                        else:
                            break
                    except ValueError:
                        print("Input invalid. Please enter an integer.")
            if i > 0 and failure_metadata[key] != '0':
                failure_comments = failure_comments + "\n" + "# " + input("Add comment: ")


        # Prompt the user to enter any additional notes
        user_notes_prompt = input("Optional -- Enter any additional notes (hit enter to continue): ")
        if not user_notes_prompt:
            user_notes = "\n" + "# Additional User Notes: None"
        else:
            user_notes = "\n" + "# Additional User Notes: " + user_notes_prompt

    # Remove newline characters from each string in questions list
    failure_questions_nonewline = [item.replace("\n   ", "") for item in failure_questions]

    # Define the preamble text
    preamble = "# This is a YAML file containing a dictionary of user responses to the following questions about the mission:\n" + "\n".join(f"#     {item}" for item in failure_questions_nonewline) + "\n"

    # Create yaml file name
    failure_fname = r"mission_failure_metadata.yaml"

    # Create the full file path
    file_path = os.path.join(datadir, failure_fname)

    # Write dictionary to YAML file
    with open(file_path, "w") as file:
        # Write the preamble text
        file.write(preamble)
        # Write the dictionary as YAML
        yaml.dump(failure_metadata, file)
        # Write the optional notes
        file.write(user_notes + "\n")
        # Write the additional comments
        file.write(failure_comments)
    print("Responses written to ", failure_fname, " in mission directory.")
