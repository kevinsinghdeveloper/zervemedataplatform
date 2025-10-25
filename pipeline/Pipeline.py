import json
import os
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Optional

from abstractions.types.enumerations.PipelineStage import PipelineStage
from abstractions.pipeline.PipelineComponent import PipelineComponent
from abstractions.types.enumerations.PipelineSubSteps import PipelineSubSteps
from utils.Utility import Utility


class PipelineUtility:
    __VARIABLE_TYPE = 'variable_type'
    __VARIABLE_VALUE = 'variable_value'

    def __init__(self, pipeline_activity_log_path):
        self.__pipeline_activity_log_path = pipeline_activity_log_path
        _ = self.get_pipeline_activity_log_contents()

    def update_pipeline_activity_log(self, content) -> None:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.__pipeline_activity_log_path), exist_ok=True)

        # Check if the checkpoint file already exists
        if os.path.exists(self.__pipeline_activity_log_path):
            with open(self.__pipeline_activity_log_path, 'r') as checkpoint_file:
                checkpoint_data = json.load(checkpoint_file)
        else:
            self.regenerate_pipeline_activity_log()

        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update or add the file entry with the current timestamp

        if checkpoint_data:
            checkpoint_data.update(content)
        else:
            checkpoint_data = content

        # Write the updated checkpoint data back to the file
        with open(self.__pipeline_activity_log_path, 'w') as pipeline_file:
            json.dump(checkpoint_data, pipeline_file, indent=4)

    def regenerate_pipeline_activity_log(self) -> None:
        # Create or overwrite the checkpoint file with an empty dictionary
        os.makedirs(os.path.dirname(self.__pipeline_activity_log_path), exist_ok=True)
        with open(self.__pipeline_activity_log_path, 'w') as pipeline_file:
            json.dump({}, pipeline_file, indent=4)

    def get_pipeline_activity_log_contents(self) -> dict:
        # Check if the checkpoint file exists
        if os.path.exists(self.__pipeline_activity_log_path):
            with open(self.__pipeline_activity_log_path, 'r') as pipeline_file:
                try:
                    pipeline_data = json.load(pipeline_file)
                    return pipeline_data  # Return a list of file names
                except json.decoder.JSONDecodeError:
                    self.regenerate_pipeline_activity_log()
                    return {}
        else:
            self.regenerate_pipeline_activity_log()
            return {}

    def add_pipeline_variable(self, task_name, variable_name, variable_type, variable_value):
        task = { task_name: {
            variable_name: {
                "variable_type": variable_type,
                "variable_value": variable_value
            }
        }}

        self.update_pipeline_activity_log(task)

    def get_pipeline_variable(self, task_name, variable_name):
        activity = self.get_pipeline_activity_log_contents()

        task = activity.get(task_name, None)

        if task is None:
            Utility.error_log(f"{task_name} not found")
            return None
            #raise Exception(f"{task_name} not found")

        variable = task.get(variable_name, None)

        if variable is None:
            Utility.error_log(f"{variable} not found")
            #raise Exception(f"{variable} not found")
            return None

        return variable.get(self.__VARIABLE_VALUE, None)

    def save_pipeline_step_progress(self, task_name: str, variable_name: str, site_name: str, identifier: str,
                                    step: PipelineSubSteps, data: dict):

        results = self.get_pipeline_variable(
            task_name=task_name,
            variable_name=variable_name)

        # Initialize results if None
        if results is None:
            results = {}

        # Check if site_name exists, and if not, initialize it
        if site_name not in results:
            results[site_name] = {}

        # Check if identifier exists under site_name, and if not, initialize it
        if identifier not in results[site_name]:
            results[site_name][identifier] = {}

        # Update the specific step with the new data
        results[site_name][identifier][step.value] = data

        # Save the updated results back to the pipeline
        self.add_pipeline_variable(
            task_name=task_name,
            variable_name=variable_name,
            variable_type="dict",
            variable_value=results
        )

    def get_pipeline_step_activity_data(self, task_name: str, variable_name: str, site_name: str, identifier: str,
                                        step: PipelineSubSteps):
        results = self.get_pipeline_variable(
            task_name=task_name,
            variable_name=variable_name)

        if not results:
            return None

        results = results.get(site_name, None)

        if not results:
            return None

        results = results.get(identifier, None)

        if not results:
            return None

        results = results.get(step.value, None)

        if not results:
            return None

        return results


class FuncPipelineStep:
    def __init__(self, name_of_step: str, func, args: dict = None):
        self.func = func
        self.args = args
        self.name_of_step = name_of_step


class FuncDataPipe(PipelineComponent):
    def __init__(self, name_of_pipe: str):
        self.name_of_pipe = name_of_pipe
        self._pipeline = []

    def add_to_pipeline(self, pipeline_step: FuncPipelineStep) -> None:
        self._pipeline.append(pipeline_step)

    def init_pipeline(self) -> None:
        self._pipeline = []

    def run_pipeline(self) -> None:
        Utility.log(f"*** Running pipeline {self.name_of_pipe}")
        count_of_steps = len(self._pipeline)
        i = 1

        for func_step in self._pipeline:
            Utility.log(f'Running STEP {func_step.name_of_step}, {i} of {count_of_steps} steps')
            try:
                if func_step.args is not None:
                    func_step.func(**func_step.args)
                else:
                    func_step.func()
            except Exception as e:
                Utility.error_log(f"Error running step {func_step.name_of_step}: {e}")
            i += 1


class DataConnectorBase(PipelineComponent, ABC):
    def __init__(self, name: str, run_datestamp: str):
        self.__pipeline_activity_logger = None
        self.__web_extractor = None
        self.name = name
        self.run_datestamp = run_datestamp

    @abstractmethod
    def run_initialize_task(self) -> None:
        pass

    @abstractmethod
    def run_pre_validate_task(self) -> None:
        pass

    @abstractmethod
    def run_read_task(self) -> None:
        pass

    @abstractmethod
    def run_main_task(self) -> None:
        pass

    @abstractmethod
    def run_output_task(self) -> None:
        pass

    def init_pipeline_activity_logger(self, pipeline_activity_log_path: str) -> None:
        self.__pipeline_activity_logger = PipelineUtility(pipeline_activity_log_path=pipeline_activity_log_path)

    def get_pipeline_activity_logger(self) -> Optional[PipelineUtility]:
        if self.__pipeline_activity_logger is None:
            Utility.warning_log("pipeline_activity_log was not initialized")
            return None
        return self.__pipeline_activity_logger

    def get_pipeline_activity_log_contents(self) -> dict:
        if self.__pipeline_activity_logger is None:
            Utility.warning_log("pipeline_activity_log was not initialized")
            return {}
        return self.__pipeline_activity_logger.get_pipeline_activity_log_contents()

    def set_pipeline_activity_log_contents(self, pipeline_activity_log_contents: dict):
        if self.__pipeline_activity_logger is None:
            Utility.warning_log("pipeline_activity_log was not initialized")
            return
        self.__pipeline_activity_logger.update_pipeline_activity_log(pipeline_activity_log_contents)


class DataPipeline(PipelineComponent):
    def __init__(self):
        self._DataPipe = []

    def init_pipeline(self) -> None:
        self._DataPipe = []

    def add_to_pipeline(self, etl_obj: DataConnectorBase) -> None:
        self._DataPipe.append(etl_obj)

    def run_only_pipeline(self, stage: PipelineStage, activity_contents: dict = None) -> dict:
        from datetime import datetime as dt
        t0 = dt.now()

        etl_obj = None
        for etl_obj in self._DataPipe:
            Utility.log(f"Running {etl_obj.name} job on stage {stage}")

            if activity_contents is not None:
                etl_obj.set_pipeline_activity_log_contents(activity_contents)

            try:
                if stage == PipelineStage.initialize_task:
                    etl_obj.run_initialize_task()
                elif stage == PipelineStage.pre_validate_task:
                    etl_obj.run_pre_validate_task()
                elif stage == PipelineStage.read_task:
                    etl_obj.run_read_task()
                elif stage == PipelineStage.main_task:
                    etl_obj.run_main_task()
                elif stage == PipelineStage.output_task:
                    etl_obj.run_output_task()
            except Exception as e:
                Utility.error_log(f"Error running {etl_obj.name} job on stage {stage}: {e}")

        t = dt.now() - t0
        Utility.log('Pipeline completed in ' + str(t))

        return etl_obj.get_pipeline_activity_log_contents()

    def run_data_pipeline(self) -> None:
        for etl_obj in self._DataPipe:
            Utility.log(f"Running {etl_obj.name} job")
            try:
                etl_obj.run_initialize_task()
                etl_obj.run_read_task()
                etl_obj.run_main_task()
                etl_obj.run_output_task()
            except Exception as e:
                Utility.error_log(f"Error running {etl_obj.name} job: {e}")

        Utility.log("PIPELINE COMPLETE")
