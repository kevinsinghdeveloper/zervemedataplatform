from typing import Dict, Union, List

import pandas as pd


class ExtensionJsonModel:
    def __init__(self):
        self.site_name = ""
        self.identifiers = {}
        self.groups = {}

    def set_site_name(self, name: str):
        self.site_name = name

    def add_site_identifier_url(self, identifier: str, url: str):
        self.identifiers[identifier] = url

    def add_selector_group(self, group_name: str):
        if group_name not in self.groups:
            self.groups[group_name] = {}

    def add_site_identifier(self, group_name: str, identifier: str):
        if group_name not in self.groups:
            self.groups[group_name] = {}
        if identifier not in self.groups[group_name]:
            self.groups[group_name][identifier] = {}

    def add_site_identifier_key(self, group_name: str, identifier: str, key: str,
                                css_selector: Union[List[str], str]):
        if group_name in self.groups and identifier in self.groups[group_name]:
            self.groups[group_name][identifier][key] = css_selector

    def generate_model_output(self) -> Dict:
        return {
            "name": self.site_name,
            "identifiers": self.identifiers,
            **self.groups
        }

    def generate_model_output_to_df(self) -> Dict[str, pd.DataFrame]:
        output = self.generate_model_output()
        dataframes = {}

        # Create a DataFrame for each identifier
        for group_name, identifiers in output.items():
            if group_name in ['name', 'identifiers']:
                continue  # Skip name and identifiers

            for identifier, keys in identifiers.items():
                # Create a new row for the DataFrame
                row = {
                    "site_name": output["name"],
                    "identifier": identifier,
                    "group": group_name
                }
                row.update(keys)  # Add the keys as columns

                # Initialize DataFrame if it does not exist
                if identifier not in dataframes:
                    dataframes[identifier] = pd.DataFrame(columns=["site_name", "identifier", "group"])

                # Concatenate the new row to the respective DataFrame
                dataframes[identifier] = pd.concat([dataframes[identifier], pd.DataFrame([row])], ignore_index=True)

        # Add URLs to each DataFrame
        for identifier, url in output["identifiers"].items():
            if identifier in dataframes:
                dataframes[identifier]['url'] = url

        return dataframes