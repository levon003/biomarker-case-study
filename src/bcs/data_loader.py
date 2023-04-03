import json
from pathlib import Path
from typing import Iterator

import pandas as pd


class DataLoader:
    def __init__(self, data_dir: Path):
        """DataLoader is a utility for loading Tempus-provided data.

        Parameters
        ----------
        data_dir : Path
            Directory to load the data from.
        """
        self.data_dir = data_dir
        self.is_data_loaded = False

    def _load(self):
        """Utility function to lazy-load the expected dataframes."""
        if self.is_data_loaded:
            return
        # load target data
        self.tdf = pd.read_csv(self.data_dir / "targets.csv")

        # load biomarker data
        self.bdf = pd.read_csv(self.data_dir / "biomarkers.csv")
        self.bdf = self.bdf.set_index("biomarker_id")

        # load patient data
        with open(self.data_dir / "patient_profiles.json") as infile:
            institution_list = json.load(infile)
        patient_profiles = DataLoader.get_patient_profiles(institution_list)
        self.pdf = pd.DataFrame(patient_profiles)
        self.pdf.set_index("patient_id")

        self.is_data_loaded = True

    def get_target_dataframe(self) -> pd.DataFrame:
        if not self.is_data_loaded:
            self._load()
        return self.tdf

    def get_biomarker_dataframe(self) -> pd.DataFrame:
        if not self.is_data_loaded:
            self._load()
        return self.bdf

    def get_patient_dataframe(self) -> pd.DataFrame:
        if not self.is_data_loaded:
            self._load()
        return self.pdf

    @staticmethod
    def get_patient_profiles(institution_list: list) -> Iterator[dict]:
        """Parses the JSON structure of the patient data.

        Parameters
        ----------
        institution_list : list
            List, as loaded from json.load().

        Yields
        ------
        Iterator[dict]
            Dictionary of flattened key-value pairs with patient data.
        """
        for institution in institution_list:
            institution_name = institution["institution"]
            cohort_id = institution["cohort_id"]
            for patient_profile in institution["patient_profiles"]:
                d = {
                    "institution_name": institution_name,
                    "cohort_id": cohort_id,
                }
                for key, value in patient_profile.items():
                    if type(value) == dict:
                        d.update({key + "_" + k: v for k, v in value.items()})
                    else:
                        d[key] = value
                yield d
