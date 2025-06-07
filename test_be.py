import unittest
from flask import Flask
from flask.testing import FlaskClient
import json

class TestMainApp(unittest.TestCase):

    def setUp(self):
        from main import app
        self.app = app
        self.client = self.app.test_client()

    def test_generate_report_route_missing_context(self):
        response = self.client.post('/generate_report', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing business context', response.get_json().get('error', ''))

    def test_generate_report_route_no_dataset(self):
        response = self.client.post('/generate_report', json={"business_context": "Financial data"})
        self.assertEqual(response.status_code, 400)
        self.assertIn('No dataset file found', response.get_json().get('error', ''))

    def test_process_query_missing_query(self):
        response = self.client.post('/process_query', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Query is missing', response.get_json().get('error', ''))

    def test_process_query_no_dataset(self):
        response = self.client.post('/process_query', json={"query": "Show me the data"})
        self.assertEqual(response.status_code, 400)
        self.assertIn('No dataset found', response.get_json().get('error', ''))

    def test_process_query_success(self):
        # Mock dataset and query for successful execution
        mock_query = "Show me the data"
        mock_column_mapping = {"col1": "Column 1", "col2": "Column 2"}
        response = self.client.post('/process_query', json={"query": mock_query, "column_mapping": mock_column_mapping})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Query executed successfully', response.get_json().get('message', ''))

if __name__ == '__main__':
    unittest.main()