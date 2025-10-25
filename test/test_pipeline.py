import unittest
import tempfile
import json
import os
import shutil
from data_platform.pipeline.Pipeline import PipelineUtility

class TestPipelineUtility(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline_activity_log_path = os.path.join(self.temp_dir, "pipeline_activity_log.json")
        self.pipeline_utility = PipelineUtility(self.pipeline_activity_log_path)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_update_pipeline_activity_log(self):
        content = {"task1": {"variable_name": {"variable_type": "type", "variable_value": "value"}}}
        self.pipeline_utility.update_pipeline_activity_log(content)

        with open(self.pipeline_activity_log_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, content)

    def test_regenerate_pipeline_activity_log(self):
        self.pipeline_utility.regenerate_pipeline_activity_log()

        with open(self.pipeline_activity_log_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, {})

    def test_get_pipeline_activity_log_contents(self):
        content = {"task1": {"variable_name": {"variable_type": "type", "variable_value": "value"}}}

        with open(self.pipeline_activity_log_path, 'w') as f:
            json.dump(content, f)

        data = self.pipeline_utility.get_pipeline_activity_log_contents()
        self.assertEqual(data, content)

    def test_add_pipeline_variable(self):
        task_name = "task1"
        variable_name = "variable_name"
        variable_type = "type"
        variable_value = "value"

        self.pipeline_utility.add_pipeline_variable(task_name, variable_name, variable_type, variable_value)

        with open(self.pipeline_activity_log_path, 'r') as f:
            data = json.load(f)
        expected_content = {
            task_name: {
                variable_name: {
                    "variable_type": variable_type,
                    "variable_value": variable_value
                }
            }
        }
        self.assertEqual(data, expected_content)

    def test_get_pipeline_variable(self):
        task_name = "task1"
        variable_name = "variable_name"
        variable_type = "type"
        variable_value = "value"
        content = {
            task_name: {
                variable_name: {
                    "variable_type": variable_type,
                    "variable_value": variable_value
                }
            }
        }

        with open(self.pipeline_activity_log_path, 'w') as f:
            json.dump(content, f)

        value = self.pipeline_utility.get_pipeline_variable(task_name, variable_name)
        self.assertEqual(value, variable_value)

    def test_get_pipeline_variable_not_found(self):
        # Test when task is not found, the method returns None
        result = self.pipeline_utility.get_pipeline_variable("task1", "variable_name")
        self.assertIsNone(result)

    def test_get_pipeline_variable_name_not_found(self):
        task_name = "task1"
        content = {task_name: {}}

        with open(self.pipeline_activity_log_path, 'w') as f:
            json.dump(content, f)

        # Test when variable name is not found, the method returns None
        result = self.pipeline_utility.get_pipeline_variable(task_name, "variable_name")
        self.assertIsNone(result)


class TestDataPipelineActivityManipulation(unittest.TestCase):
    """Test activity content manipulation across pipeline stages"""

    def setUp(self):
        """Set up test fixtures with a mock ETL component"""
        from data_platform.pipeline.Pipeline import DataPipeline, DataConnectorBase

        # Create a temporary directory for the pipeline activity log
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline_activity_log_path = os.path.join(self.temp_dir, "test_pipeline_activity.json")

        # Create a mock ETL component
        class MockETLComponent(DataConnectorBase):
            def __init__(self, name, run_datestamp):
                super().__init__(name, run_datestamp)
                self.initialize_called = False
                self.pre_validate_called = False
                self.read_called = False
                self.main_called = False
                self.output_called = False

            def run_initialize_task(self):
                self.initialize_called = True
                # Add some initial data to the activity log
                self.get_pipeline_activity_logger().add_pipeline_variable(
                    task_name="initialize_task",
                    variable_name="init_status",
                    variable_type="str",
                    variable_value="initialized"
                )

            def run_pre_validate_task(self):
                self.pre_validate_called = True
                # Read a variable from activity log
                init_status = self.get_pipeline_activity_logger().get_pipeline_variable(
                    task_name="initialize_task",
                    variable_name="init_status"
                )
                # Add pre-validation data
                self.get_pipeline_activity_logger().add_pipeline_variable(
                    task_name="pre_validate_task",
                    variable_name="validation_status",
                    variable_type="str",
                    variable_value=f"validated after {init_status}"
                )

            def run_read_task(self):
                self.read_called = True

            def run_main_task(self):
                self.main_called = True

            def run_output_task(self):
                self.output_called = True

        self.mock_etl = MockETLComponent("TestETL", "20250122")
        self.mock_etl.init_pipeline_activity_logger(self.pipeline_activity_log_path)

        self.data_pipeline = DataPipeline()
        self.data_pipeline.add_to_pipeline(self.mock_etl)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_activity_content_return_and_manipulation(self):
        """Test returning and manipulating activity content across pipeline stages"""
        from data_platform.abstractions.types.enumerations.PipelineStage import PipelineStage

        # Step 1: Run initialize task and get activity contents
        activity_contents = self.data_pipeline.run_only_pipeline(PipelineStage.initialize_task)

        # Verify initialize task was called
        self.assertTrue(self.mock_etl.initialize_called)

        # Verify activity_contents has the initialized data
        self.assertIn("initialize_task", activity_contents)
        self.assertIn("init_status", activity_contents["initialize_task"])
        self.assertEqual(
            activity_contents["initialize_task"]["init_status"]["variable_value"],
            "initialized"
        )

        # Step 2: Manually manipulate activity_contents (as shown in run_prod_data_collector.py)
        activity_contents['test'] = {
            'variable_type': 'str',
            'variable_value': 'test item'
        }

        # Verify the manual addition
        self.assertIn('test', activity_contents)
        self.assertEqual(activity_contents['test']['variable_value'], 'test item')

        # Step 3: Pass modified activity_contents to the next stage
        activity_contents2 = self.data_pipeline.run_only_pipeline(
            PipelineStage.pre_validate_task,
            activity_contents
        )

        # Verify pre_validate task was called
        self.assertTrue(self.mock_etl.pre_validate_called)

        # Verify activity_contents2 contains both:
        # - Original initialize_task data
        # - Manually added 'test' data
        # - New pre_validate_task data
        self.assertIn("initialize_task", activity_contents2)
        self.assertIn("test", activity_contents2)
        self.assertIn("pre_validate_task", activity_contents2)

        # Verify the test data persisted through the pipeline
        self.assertEqual(activity_contents2['test']['variable_value'], 'test item')

        # Verify the pre_validate task processed the data
        self.assertEqual(
            activity_contents2["pre_validate_task"]["validation_status"]["variable_value"],
            "validated after initialized"
        )

    def test_activity_content_chaining_multiple_stages(self):
        """Test chaining activity contents through multiple pipeline stages"""
        from data_platform.abstractions.types.enumerations.PipelineStage import PipelineStage

        # Chain 1: Initialize
        activity_contents = self.data_pipeline.run_only_pipeline(PipelineStage.initialize_task)

        # Chain 2: Add custom data
        activity_contents['custom_config'] = {
            'variable_type': 'dict',
            'variable_value': {'timeout': 30, 'retries': 3}
        }

        # Chain 3: Pre-validate with modified contents
        activity_contents = self.data_pipeline.run_only_pipeline(
            PipelineStage.pre_validate_task,
            activity_contents
        )

        # Verify all data is preserved
        self.assertIn('initialize_task', activity_contents)
        self.assertIn('custom_config', activity_contents)
        self.assertIn('pre_validate_task', activity_contents)

        # Verify custom config is intact
        self.assertEqual(
            activity_contents['custom_config']['variable_value']['timeout'],
            30
        )

    def test_activity_content_without_passing_to_next_stage(self):
        """Test that not passing activity_contents starts fresh in next stage"""
        from data_platform.abstractions.types.enumerations.PipelineStage import PipelineStage

        # Run initialize task
        activity_contents = self.data_pipeline.run_only_pipeline(PipelineStage.initialize_task)

        # Manually add test data
        activity_contents['test_data'] = {
            'variable_type': 'str',
            'variable_value': 'should not persist'
        }

        # Run pre_validate WITHOUT passing activity_contents (i.e., passing None)
        activity_contents2 = self.data_pipeline.run_only_pipeline(
            PipelineStage.pre_validate_task,
            None  # Not passing the modified contents
        )

        # The manually added 'test_data' should NOT be in activity_contents2
        # because we didn't pass it to the next stage
        # However, both tasks' data should be in the log because they write to the same file
        self.assertIn('initialize_task', activity_contents2)
        self.assertIn('pre_validate_task', activity_contents2)
        # The test_data was never written to the log, so it won't be there
        self.assertNotIn('test_data', activity_contents2)


if __name__ == '__main__':
    unittest.main()