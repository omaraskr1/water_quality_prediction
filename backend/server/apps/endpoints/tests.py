from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
        input_data = {
            "ph": 5.735724,
            "Hardness": 158.318741,
            "Solids": 25363.016594,
            "Chloramines": 7.728601,
            "Sulfate": 377.543291,
            "Conductivity": 568.304671,
            "Organic_carbon": 13.626624,
            "Trihalomethanes": 75.952337,
            "Turbidity": 4.732954
           
        }
        classifier_url = "/api/v1/waterqualilty_classifierr/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], "0")
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)